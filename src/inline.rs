use core::{
    hash::{BuildHasher, Hash},
    iter::FusedIterator,
    mem::{self, transmute, MaybeUninit},
    ptr::NonNull,
};

use crate::{
    raw::{
        h2,
        iter::{RawIntoIter, RawIter},
        util::{equivalent_key, likely, make_hash, EntryRef, SizedTypeProperties, VacantEntry},
        BitMaskWord, Group, RawIterInner, DELETED, EMPTY,
    },
    Equivalent,
};

#[derive(Clone)]
pub struct Inline<const N: usize, K, V, S> {
    raw: RawInline<N, (K, V)>,
    // Option is for take, S always exists before drop.
    hash_builder: Option<S>,
}

struct RawInline<const N: usize, T> {
    aligned_tags: AlignedTags<N>,
    len: usize,
    entries: [MaybeUninit<T>; N],
}

impl<const N: usize, T: Clone> Clone for RawInline<N, T> {
    #[inline]
    fn clone(&self) -> Self {
        let mut data = unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() };
        for (idx, d) in self.entries.iter().take(self.len).enumerate() {
            unsafe {
                data[idx] = MaybeUninit::new(d.assume_init_ref().clone());
            }
        }
        Self {
            aligned_tags: self.aligned_tags,
            len: self.len,
            entries: data,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct AlignedTags<const N: usize> {
    tags: [u8; N],
    _align: [Group; 0],
}

impl<const N: usize> AlignedTags<N> {
    #[inline]
    unsafe fn tag(&self, index: usize) -> *mut u8 {
        self.tags.as_ptr().add(index).cast_mut()
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> NonNull<u8> {
        unsafe { NonNull::new_unchecked(self.tags.as_ptr() as _) }
    }

    #[inline]
    pub(crate) unsafe fn group_unchecked(&self, index: usize) -> Group {
        let tag_index = index * Group::SIZE;
        Group::load(self.tag(tag_index))
    }

    #[inline]
    pub(crate) fn num_complete_groups(&self) -> usize {
        Self::NUM_GROUPS
    }
    const NUM_GROUPS: usize = N / Group::SIZE;

    #[inline]
    pub(crate) fn tail_mask(&self) -> Option<BitMaskWord> {
        if N % Group::SIZE == 0 {
            return None;
        }
        Some(Self::TAIL_MASK)
    }
    const TAIL_MASK: BitMaskWord = Group::LOWEST_MASK[N % Group::SIZE];

    #[inline]
    pub(crate) fn group_size(&self) -> usize {
        Group::SIZE
    }
}

impl<const N: usize, T> Drop for RawInline<N, T> {
    #[inline]
    fn drop(&mut self) {
        unsafe { self.drop_elements() }
    }
}

impl<const N: usize, T> RawInline<N, T> {
    #[inline]
    unsafe fn drop_elements(&mut self) {
        if T::NEEDS_DROP && self.len != 0 {
            unsafe {
                drop(RawIntoIter {
                    inner: self.raw_iter_inner(),
                    aligned_tags: (&self.aligned_tags as *const AlignedTags<N>).read(),
                    entries: (&self.entries as *const [MaybeUninit<T>; N]).read(),
                });
            }
        }
    }

    /// Gets a reference to an element in the table.
    #[inline]
    fn get(&self, hash: u64, matches_entry: impl FnMut(&T) -> bool) -> Option<&T> {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.find(hash, matches_entry) {
            Some(bucket) => Some(unsafe { bucket.as_ref() }),
            None => None,
        }
    }

    /// Gets a mutable reference to an element in the table.
    #[inline]
    fn get_mut(&mut self, hash: u64, matches_entry: impl FnMut(&T) -> bool) -> Option<&mut T> {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.find(hash, matches_entry) {
            Some(bucket) => Some(unsafe { bucket.as_mut() }),
            None => None,
        }
    }

    /// Searches for an element in the table.
    #[inline]
    fn find(
        &self,
        value_hash: u64,
        mut matches_entry: impl FnMut(&T) -> bool,
    ) -> Option<EntryRef<T>> {
        unsafe {
            let value_hash = h2(value_hash);
            let mut group_index = 0;

            // Manually expand the loop
            for _ in 0..self.aligned_tags.num_complete_groups() {
                let group = self.aligned_tags.group_unchecked(group_index);
                let matched_tags = group.match_byte(value_hash);

                for tag_index in matched_tags {
                    let entry_index = self.entry_index(group_index, tag_index);
                    if likely(matches_entry(
                        self.entries.get_unchecked(entry_index).assume_init_ref(),
                    )) {
                        return Some(self.bucket(entry_index));
                    }
                }
                group_index += 1;
            }
            if let Some(tail_mask) = self.aligned_tags.tail_mask() {
                let group = self.aligned_tags.group_unchecked(group_index);
                // Clear invalid tail.
                let matched_tags = group.match_byte(value_hash).and(tail_mask);

                for tag_index in matched_tags {
                    let entry_index = self.entry_index(group_index, tag_index);
                    if likely(matches_entry(
                        self.entries.get_unchecked(entry_index).assume_init_ref(),
                    )) {
                        return Some(self.bucket(entry_index));
                    }
                }
            }
            None
        }
    }

    /// Searches for an element in the table. If the element is not found,
    /// returns `Err` with the position of a slot where an element with the
    /// same hash could be inserted.
    #[inline]
    fn find_or_find_vacant_entry(
        &mut self,
        hash: u64,
        mut matches_entry: impl FnMut(&T) -> bool,
    ) -> Result<EntryRef<T>, VacantEntry> {
        unsafe {
            let mut vacant_entry = None;
            let tag = h2(hash);
            let mut group_index = 0;

            // Manually expand the loop
            for _ in 0..self.aligned_tags.num_complete_groups() {
                let group = self.aligned_tags.group_unchecked(group_index);
                let matched_tags = group.match_byte(tag);
                for tag_index in matched_tags {
                    let entry_index = self.entry_index(group_index, tag_index);
                    if likely(matches_entry(
                        self.entries.get_unchecked(entry_index).assume_init_ref(),
                    )) {
                        return Ok(self.bucket(entry_index));
                    }
                }

                // We didn't find the element we were looking for in the group, try to get an
                // insertion slot from the group if we don't have one yet.
                if likely(vacant_entry.is_none()) {
                    vacant_entry = self.find_vacant_entry_in_group(&group, group_index);
                }

                // If there's empty set, we should stop searching next group.
                if likely(group.match_empty().any_bit_set()) {
                    break;
                }
                group_index += 1;
            }
            if let Some(tail_mask) = self.aligned_tags.tail_mask() {
                let group = self.aligned_tags.group_unchecked(group_index);
                let matched_tags = group.match_byte(tag).and(tail_mask);
                for tag_index in matched_tags {
                    let entry_index = self.entry_index(group_index, tag_index);
                    if likely(matches_entry(
                        self.entries.get_unchecked(entry_index).assume_init_ref(),
                    )) {
                        return Ok(self.bucket(entry_index));
                    }
                }

                // We didn't find the element we were looking for in the group, try to get an
                // insertion slot from the group if we don't have one yet.
                if likely(vacant_entry.is_none()) {
                    vacant_entry = self.find_vacant_entry_in_group(&group, group_index);
                }
            }

            Err(VacantEntry {
                index: vacant_entry.unwrap_unchecked(),
            })
        }
    }

    /// Finds the position to insert something in a group.
    #[inline]
    fn find_vacant_entry_in_group(&self, group: &Group, group_index: usize) -> Option<usize> {
        let invalid_tag_index = group.match_empty_or_deleted().lowest_set_bit();

        if likely(invalid_tag_index.is_some()) {
            let invalid_tag_index = unsafe { invalid_tag_index.unwrap_unchecked() };
            return Some(self.entry_index(group_index, invalid_tag_index));
        }
        None
    }

    /// Inserts a new element into the table in the given slot, and returns its
    /// raw bucket.
    #[inline]
    unsafe fn insert_to_vacant_entry(
        &mut self,
        hash: u64,
        slot: VacantEntry,
        value: T,
    ) -> EntryRef<T> {
        self.record_item_insert_at(slot.index, hash);
        let bucket = self.bucket(slot.index);
        bucket.write(value);
        bucket
    }

    /// Inserts a new element into the table in the given slot, and returns its
    /// raw bucket.
    #[inline]
    unsafe fn record_item_insert_at(&mut self, index: usize, hash: u64) {
        self.set_tag_h2(index, hash);
        self.len += 1;
    }

    /// Sets a control byte to the hash, and possibly also the replicated control byte at
    /// the end of the array.
    #[inline]
    unsafe fn set_tag_h2(&mut self, index: usize, hash: u64) {
        // SAFETY: The caller must uphold the safety rules for the [`RawTableInner::set_ctrl_h2`]
        *self.aligned_tags.tag(index) = h2(hash);
    }

    /// Finds and removes an element from the table, returning it.
    #[inline]
    fn remove_entry(&mut self, hash: u64, matches_entry: impl FnMut(&T) -> bool) -> Option<T> {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.find(hash, matches_entry) {
            Some(bucket) => Some(unsafe { self.remove(bucket).0 }),
            None => None,
        }
    }

    /// Removes an element from the table, returning it.
    #[inline]
    #[allow(clippy::needless_pass_by_value)]
    unsafe fn remove(&mut self, item: EntryRef<T>) -> (T, VacantEntry) {
        self.erase_no_drop(&item);
        (
            item.read(),
            VacantEntry {
                index: self.bucket_index(&item),
            },
        )
    }

    /// Erases an element from the table without dropping it.
    #[inline]
    unsafe fn erase_no_drop(&mut self, item: &EntryRef<T>) {
        let index = self.bucket_index(item);
        self.erase(index);
    }

    /// Returns the index of a bucket from a `Bucket`.
    #[inline]
    unsafe fn bucket_index(&self, bucket: &EntryRef<T>) -> usize {
        bucket.to_base_index(NonNull::new_unchecked(self.entries.as_ptr() as _))
    }

    /// Erases the [`Bucket`]'s control byte at the given index so that it does not
    /// triggered as full, decreases the `items` of the table and, if it can be done,
    /// increases `self.growth_left`.
    #[inline]
    unsafe fn erase(&mut self, entry_index: usize) {
        *self.aligned_tags.tag(entry_index) = DELETED;
        self.len -= 1;
    }

    /// Returns a pointer to an element in the table.
    #[inline]
    unsafe fn bucket(&self, entry_index: usize) -> EntryRef<T> {
        EntryRef::from_base_index(
            NonNull::new_unchecked(transmute(self.entries.as_ptr().cast_mut())),
            entry_index,
        )
    }

    #[inline]
    unsafe fn raw_iter_inner(&self) -> RawIterInner<T> {
        let init_group = Group::load_aligned(self.aligned_tags.tag(0)).match_full();
        RawIterInner::new(init_group, self.len)
    }

    #[inline]
    fn iter(&self) -> RawIter<'_, N, T> {
        RawIter {
            inner: unsafe { self.raw_iter_inner() },
            aligned_groups: &self.aligned_tags,
            data: &self.entries,
        }
    }

    fn entry_index(&self, group_index: usize, tag_index: usize) -> usize {
        self.aligned_tags.group_size() * group_index + tag_index
    }
}

impl<const N: usize, T> IntoIterator for RawInline<N, T> {
    type Item = T;
    type IntoIter = RawIntoIter<N, T>;

    #[inline]
    fn into_iter(self) -> RawIntoIter<N, T> {
        let ret = unsafe {
            RawIntoIter {
                inner: self.raw_iter_inner(),
                aligned_tags: (&self.aligned_tags as *const AlignedTags<N>).read(),
                entries: (&self.entries as *const [MaybeUninit<T>; N]).read(),
            }
        };
        mem::forget(self);
        ret
    }
}

pub struct Iter<'a, const N: usize, K, V> {
    inner: RawIter<'a, N, (K, V)>,
}

pub struct IntoIter<const N: usize, K, V> {
    inner: RawIntoIter<N, (K, V)>,
}

impl<'a, const N: usize, K, V> Iterator for Iter<'a, N, K, V> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        match self.inner.next() {
            Some(kv) => Some((&kv.0, &kv.1)),
            None => None,
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<const N: usize, K, V> Iterator for IntoIter<N, K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<(K, V)> {
        self.inner.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
impl<'a, const N: usize, K, V> ExactSizeIterator for Iter<'a, N, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}
impl<const N: usize, K, V> ExactSizeIterator for IntoIter<N, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}
impl<'a, const N: usize, K, V> FusedIterator for Iter<'a, N, K, V> {}
impl<const N: usize, K, V> FusedIterator for IntoIter<N, K, V> {}

impl<const N: usize, K, V, S> IntoIterator for Inline<N, K, V, S> {
    type Item = (K, V);
    type IntoIter = IntoIter<N, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.raw.into_iter(),
        }
    }
}

impl<const N: usize, K, V, S> Inline<N, K, V, S> {
    #[inline]
    pub(crate) fn iter(&self) -> Iter<'_, N, K, V> {
        Iter {
            inner: self.raw.iter(),
        }
    }

    #[inline]
    pub(crate) const fn new(hash_builder: S) -> Self {
        assert!(N != 0, "SmallMap cannot be initialized with zero size.");
        Self {
            raw: RawInline {
                aligned_tags: AlignedTags {
                    tags: [EMPTY; N],
                    _align: [],
                },
                len: 0,
                // TODO: use uninit_array when stable
                entries: unsafe { MaybeUninit::<[MaybeUninit<(K, V)>; N]>::uninit().assume_init() },
            },
            hash_builder: Some(hash_builder),
        }
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.raw.len == 0
    }

    #[inline]
    pub(crate) fn is_full(&self) -> bool {
        self.raw.len == N
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.raw.len
    }

    // # Safety
    // Hasher must exist.
    #[inline]
    pub(crate) unsafe fn take_hasher(&mut self) -> S {
        self.hash_builder.take().unwrap_unchecked()
    }

    #[inline]
    fn hash_builder(&self) -> &S {
        self.hash_builder.as_ref().unwrap()
    }
}

impl<const N: usize, K, V, S> Inline<N, K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Returns a reference to the value corresponding to the key.
    #[inline]
    pub(crate) fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.get_inner(k) {
            Some((_, v)) => Some(v),
            None => None,
        }
    }

    /// Returns a reference to the value corresponding to the key.
    #[inline]
    pub(crate) fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.get_inner_mut(k) {
            Some((_, v)) => Some(v),
            None => None,
        }
    }

    /// Returns the key-value pair corresponding to the supplied key.
    #[inline]
    pub(crate) fn get_key_value<Q>(&self, k: &Q) -> Option<(&K, &V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.get_inner(k) {
            Some((key, value)) => Some((key, value)),
            None => None,
        }
    }

    /// Inserts a key-value pair into the map.
    #[inline]
    pub(crate) fn insert(&mut self, k: K, v: V) -> Option<V> {
        let hash = make_hash::<K, S>(self.hash_builder(), &k);
        match self.raw.find_or_find_vacant_entry(hash, equivalent_key(&k)) {
            Ok(bucket) => Some(mem::replace(unsafe { &mut bucket.as_mut().1 }, v)),
            Err(slot) => {
                unsafe {
                    self.raw.insert_to_vacant_entry(hash, slot, (k, v));
                }
                None
            }
        }
    }

    /// Removes a key from the map, returning the stored key and value if the
    /// key was previously in the map. Keeps the allocated memory for reuse.
    #[inline]
    pub(crate) fn remove_entry<Q>(&mut self, k: &Q) -> Option<(K, V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = make_hash::<Q, S>(self.hash_builder(), k);
        self.raw.remove_entry(hash, equivalent_key(k))
    }

    #[inline]
    fn get_inner<Q>(&self, k: &Q) -> Option<&(K, V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        if self.is_empty() {
            None
        } else {
            let hash = make_hash::<Q, S>(self.hash_builder(), k);
            self.raw.get(hash, equivalent_key(k))
        }
    }

    #[inline]
    fn get_inner_mut<Q>(&mut self, k: &Q) -> Option<&mut (K, V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        if self.is_empty() {
            None
        } else {
            let hash = make_hash::<Q, S>(self.hash_builder(), k);
            self.raw.get_mut(hash, equivalent_key(k))
        }
    }
}
