#[cfg(test)]
mod tests {
    use backend_tests_derive::generate_backend_tests;
    use burn::tensor::{
        backend::Backend,
        Shape,
        Tensor,
    };

    #[generate_backend_tests]
    fn test_dims<B: Backend>(device: B::Device) {
        let tensor: Tensor<B, 2> = Tensor::ones([2, 2], &device);
        assert_eq!(tensor.dims(), [2, 2]);
    }
    #[generate_backend_tests]
    fn test_shape<B: Backend>(device: B::Device) {
        let tensor: Tensor<B, 2> = Tensor::ones([2, 2], &device);

        assert_eq!(tensor.shape(), Shape::new([2, 2]));
        assert_eq!(tensor.shape().dims(), [2, 2]);
        assert_eq!(tensor.shape().dims, vec![2, 2]);

        assert_eq!(tensor.shape().num_dims(), 2);
        assert_eq!(tensor.shape().num_elements(), 4);
    }
}
