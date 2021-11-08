pub trait ToBytes {
    fn to_bytes(&self) -> &[u8];
}

impl<T> ToBytes for Vec<T> where T: bytemuck::Pod {
    fn to_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self[..])
    }
}

impl<T, const N: usize> ToBytes for [T; N] where T: bytemuck::Pod {
    fn to_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}
