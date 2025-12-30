use std::{
    iter::Zip,
    ops::{Index, IndexMut},
    slice::{Iter, IterMut},
    usize,
};

use thiserror::Error;

use crate::math::{Quaternion, Transform3, Vector3};

#[derive(Clone)]
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
        self.id_to_index.resize(id + 1, usize::MAX);

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
        if id >= self.id_to_index.len() || self.id_to_index[id] == usize::MAX {
            return None;
        }

        Some(&self.data[self.id_to_index[id]])
    }

    pub fn get_mut(&mut self, id: usize) -> Option<&mut T> {
        if id >= self.id_to_index.len() || self.id_to_index[id] == usize::MAX {
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
pub struct IdReserver {
    id: usize,
    deleted: Vec<usize>,
}

impl IdReserver {
    pub fn reserve(&mut self) -> usize {
        if let Some(id) = self.deleted.pop() {
            return id;
        }

        let id = self.id;
        self.id += 1;

        id
    }

    pub fn delete(&mut self, id: usize) {
        self.deleted.push(id);
    }
}

#[derive(Debug, Error)]
pub enum ByteStreamError {
    #[error("Out of bounds writer/read index")]
    OutOfBounds,
}

#[derive(Default)]
pub struct ByteStream<'a> {
    buffer: &'a mut [u8],
    read: usize,
    write: usize,
}

impl<'a> ByteStream<'a> {
    pub fn new(buffer: &'a mut [u8]) -> Self {
        Self {
            buffer,
            read: 0,
            write: 0,
        }
    }

    pub fn seek_write(&mut self, cursor: usize) -> Result<(), ByteStreamError> {
        if cursor >= self.buffer.len() {
            return Err(ByteStreamError::OutOfBounds);
        }

        self.write = cursor;

        Ok(())
    }

    pub fn seek_read(&mut self, cursor: usize) -> Result<(), ByteStreamError> {
        if cursor >= self.buffer.len() {
            return Err(ByteStreamError::OutOfBounds);
        }

        self.read = cursor;

        Ok(())
    }

    pub fn get_write(&self) -> usize {
        self.write
    }

    pub fn get_read(&self) -> usize {
        self.read
    }

    pub fn write_u8(&mut self, data: u8) -> Result<(), ByteStreamError> {
        if self.write == self.buffer.len() {
            return Err(ByteStreamError::OutOfBounds);
        }

        self.buffer[self.write] = data;
        self.write += 1;

        Ok(())
    }

    pub fn write_u16(&mut self, data: u16) -> Result<(), ByteStreamError> {
        if self.write + 1 >= self.buffer.len() {
            return Err(ByteStreamError::OutOfBounds);
        }

        for i in 0..2 {
            self.buffer[self.write] = data.to_be_bytes()[i];
            self.write += 1;
        }

        Ok(())
    }

    pub fn write_u32(&mut self, data: u32) -> Result<(), ByteStreamError> {
        if self.write + 3 >= self.buffer.len() {
            return Err(ByteStreamError::OutOfBounds);
        }

        for i in 0..4 {
            self.buffer[self.write] = data.to_be_bytes()[i];
            self.write += 1;
        }

        Ok(())
    }

    pub fn write_u64(&mut self, data: u64) -> Result<(), ByteStreamError> {
        if self.write + 7 >= self.buffer.len() {
            return Err(ByteStreamError::OutOfBounds);
        }

        for i in 0..8 {
            self.buffer[self.write] = data.to_be_bytes()[i];
            self.write += 1;
        }

        Ok(())
    }

    pub fn write_i8(&mut self, data: i8) -> Result<(), ByteStreamError> {
        self.write_u8(data as u8)
    }

    pub fn write_i16(&mut self, data: i16) -> Result<(), ByteStreamError> {
        self.write_u16(data as u16)
    }

    pub fn write_i32(&mut self, data: i32) -> Result<(), ByteStreamError> {
        self.write_u32(data as u32)
    }

    pub fn write_i64(&mut self, data: i64) -> Result<(), ByteStreamError> {
        self.write_u64(data as u64)
    }

    pub fn write_f32(&mut self, data: f32) -> Result<(), ByteStreamError> {
        if self.write + 3 >= self.buffer.len() {
            return Err(ByteStreamError::OutOfBounds);
        }

        for i in 0..4 {
            self.buffer[self.write] = data.to_be_bytes()[i];
            self.write += 1;
        }

        Ok(())
    }
    pub fn write_f64(&mut self, data: f64) -> Result<(), ByteStreamError> {
        if self.write + 7 >= self.buffer.len() {
            return Err(ByteStreamError::OutOfBounds);
        }

        for i in 0..8 {
            self.buffer[self.write] = data.to_be_bytes()[i];
            self.write += 1;
        }

        Ok(())
    }

    pub fn write_vec3(&mut self, data: Vector3) -> Result<(), ByteStreamError> {
        self.write_f32(data.x)?;
        self.write_f32(data.y)?;
        self.write_f32(data.z)?;

        Ok(())
    }

    pub fn write_quaternion(&mut self, data: Quaternion) -> Result<(), ByteStreamError> {
        self.write_f32(data.x)?;
        self.write_f32(data.y)?;
        self.write_f32(data.z)?;
        self.write_f32(data.w)?;

        Ok(())
    }

    pub fn write_transform3(&mut self, data: Transform3) -> Result<(), ByteStreamError> {
        self.write_vec3(data.position)?;
        self.write_quaternion(data.rotation)?;

        Ok(())
    }

    pub fn read_u8(&mut self) -> Result<u8, ByteStreamError> {
        if self.read == self.buffer.len() {
            return Err(ByteStreamError::OutOfBounds);
        }

        let data = self.buffer[self.read];
        self.read += 1;

        Ok(data)
    }
    pub fn read_u16(&mut self) -> Result<u16, ByteStreamError> {
        if self.read + 1 >= self.buffer.len() {
            return Err(ByteStreamError::OutOfBounds);
        }

        let mut data = [0_u8; 2];

        for i in 0..2 {
            data[i] = self.buffer[self.read];
            self.read += 1;
        }

        Ok(u16::from_be_bytes(data))
    }
    pub fn read_u32(&mut self) -> Result<u32, ByteStreamError> {
        if self.read + 3 >= self.buffer.len() {
            return Err(ByteStreamError::OutOfBounds);
        }

        let mut data = [0_u8; 4];

        for i in 0..4 {
            data[i] = self.buffer[self.read];
            self.read += 1;
        }

        Ok(u32::from_be_bytes(data))
    }
    pub fn read_u64(&mut self) -> Result<u64, ByteStreamError> {
        if self.read + 7 >= self.buffer.len() {
            return Err(ByteStreamError::OutOfBounds);
        }

        let mut data = [0_u8; 8];

        for i in 0..8 {
            data[i] = self.buffer[self.read];
            self.read += 1;
        }

        Ok(u64::from_be_bytes(data))
    }

    pub fn read_i8(&mut self) -> Result<i8, ByteStreamError> {
        Ok(self.read_u8()? as i8)
    }

    pub fn read_i16(&mut self) -> Result<i16, ByteStreamError> {
        Ok(self.read_u16()? as i16)
    }

    pub fn read_i32(&mut self) -> Result<i32, ByteStreamError> {
        Ok(self.read_u32()? as i32)
    }

    pub fn read_i64(&mut self) -> Result<i64, ByteStreamError> {
        Ok(self.read_u64()? as i64)
    }

    pub fn read_f32(&mut self) -> Result<f32, ByteStreamError> {
        if self.read + 3 >= self.buffer.len() {
            return Err(ByteStreamError::OutOfBounds);
        }

        let mut data = [0_u8; 4];

        for i in 0..4 {
            data[i] = self.buffer[self.read];
            self.read += 1;
        }

        Ok(f32::from_be_bytes(data))
    }

    pub fn read_f64(&mut self) -> Result<f64, ByteStreamError> {
        if self.read + 3 >= self.buffer.len() {
            return Err(ByteStreamError::OutOfBounds);
        }

        let mut data = [0_u8; 8];

        for i in 0..8 {
            data[i] = self.buffer[self.read];
            self.read += 1;
        }

        Ok(f64::from_be_bytes(data))
    }

    pub fn read_vec3(&mut self) -> Result<Vector3, ByteStreamError> {
        Ok(Vector3::new(
            self.read_f32()?,
            self.read_f32()?,
            self.read_f32()?,
        ))
    }

    pub fn read_quaternion(&mut self) -> Result<Quaternion, ByteStreamError> {
        Ok(Quaternion::new(
            self.read_f32()?,
            self.read_f32()?,
            self.read_f32()?,
            self.read_f32()?,
        ))
    }

    pub fn read_transform3(&mut self) -> Result<Transform3, ByteStreamError> {
        Ok(Transform3::new(self.read_vec3()?, self.read_quaternion()?))
    }
}
