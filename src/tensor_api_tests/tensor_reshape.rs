#[cfg(test)]
mod tests {
    use backend_tests_derive::generate_backend_tests;
    use burn::tensor::{
        backend::Backend,
        Shape,
        Tensor,
    };

    #[generate_backend_tests]
    fn test_reshape<B: Backend>(device: B::Device) {
        let tensor: Tensor<B, 3> = Tensor::ones([2, 3, 4], &device);

        let tensor = tensor.reshape([2, -1]);
        assert_eq!(tensor.shape(), Shape::new([2, 12]));

        let tensor = tensor.reshape([2, 1, 6, 2]);
        assert_eq!(tensor.shape(), Shape::new([2, 1, 6, 2]));
    }
}
