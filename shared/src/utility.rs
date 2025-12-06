use std::{
    iter::Zip,
    ops::{Index, IndexMut},
    slice::{Iter, IterMut},
};

pub struct SparseSet<T> {
    id_to_index: Vec<usize>,
    index_to_id: Vec<usize>,

    data: Vec<T>,
}

impl<T> Default for SparseSet<T> {
    fn default() -> Self {
        Self {
            id_to_index: Default::default(),
            index_to_id: Default::default(),
            data: Default::default(),
        }
    }
}

impl<T> SparseSet<T> {
    pub fn insert(&mut self, id: usize, data: T) {
        self.id_to_index.resize(id + 1, 0);

        let index = self.index_to_id.len();
        self.id_to_index[id] = index;

        self.index_to_id.push(id);
        self.data.push(data);
    }
    pub fn delete(&mut self, id: usize) -> usize {
        let index = self.id_to_index[id];
        let replacement_index = self.index_to_id.len() - 1;

        if let Some(last) = self.data.pop() {
            self.data[index] = last;
        }

        self.id_to_index[self.index_to_id[replacement_index]] = index;
        index
    }

    pub fn iter<'a>(&'a self) -> SparseSetIter<'a, T> {
        SparseSetIter::new(self)
    }

    pub fn iter_mut<'a>(&'a mut self) -> SparseSetIterMut<'a, T> {
        SparseSetIterMut::new(self)
    }

    pub fn get(&self, id: usize) -> Option<&T> {
        if id > self.id_to_index.len() || self.id_to_index[id] == usize::MAX {
            return None;
        }

        Some(&self.data[self.id_to_index[id]])
    }

    pub fn get_mut(&mut self, id: usize) -> Option<&mut T> {
        if id > self.id_to_index.len() || self.id_to_index[id] == usize::MAX {
            return None;
        }

        Some(&mut self.data[self.id_to_index[id]])
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl<T> Index<usize> for SparseSet<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.id_to_index[index]]
    }
}

impl<T> IndexMut<usize> for SparseSet<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[self.id_to_index[index]]
    }
}

pub struct SparseSetIter<'a, T> {
    iter: Zip<Iter<'a, usize>, Iter<'a, T>>,
}

impl<'a, T> SparseSetIter<'a, T> {
    pub fn new(spares_set: &'a SparseSet<T>) -> Self {
        Self {
            iter: spares_set.index_to_id.iter().zip(spares_set.data.iter()),
        }
    }
}

impl<'a, T> Iterator for SparseSetIter<'a, T> {
    type Item = (&'a usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub struct SparseSetIterMut<'a, T> {
    iter: Zip<Iter<'a, usize>, IterMut<'a, T>>,
}

impl<'a, T> SparseSetIterMut<'a, T> {
    pub fn new(spares_set: &'a mut SparseSet<T>) -> Self {
        Self {
            iter: spares_set
                .index_to_id
                .iter()
                .zip(spares_set.data.iter_mut()),
        }
    }
}

impl<'a, T> Iterator for SparseSetIterMut<'a, T> {
    type Item = (&'a usize, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

#[derive(Default)]
pub struct EntityReserver {
    id: usize,
}

impl EntityReserver {
    pub fn reserve(&mut self) -> usize {
        let id = self.id;
        self.id += 1;

        id
    }
}
