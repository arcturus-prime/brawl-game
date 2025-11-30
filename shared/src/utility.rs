pub struct SparseSetIndexer {
    id_to_index: Vec<usize>,
    index_to_id: Vec<usize>,
}

impl SparseSetIndexer {
    pub fn reserve(&mut self, id: usize) {
        self.id_to_index.resize(id, 0);

        let index = self.index_to_id.len();
        self.id_to_index[id] = index;

        self.index_to_id.push(id);
    }
    pub fn delete(&mut self, id: usize) -> usize {
        let index = self.id_to_index[id];
        let replacement_index = self.colliders.len() - 1;

        self.id_to_index[self.index_to_id[replacement_index]] = index;
        index
    }

    pub fn get(&self, id: usize) -> usize {
        self.id_to_index[id]
    }
}
