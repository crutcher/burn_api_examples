#[cfg(test)]
mod tests {
    use backend_tests_derive::generate_backend_tests;
    use burn::tensor::{
        backend::Backend,
        Tensor,
    };

    #[generate_backend_tests]
    fn test_device<B: Backend>(device: B::Device) {
        let tensor: Tensor<B, 2> = Tensor::ones([2, 2], &device);
        assert_eq!(tensor.device(), device);
    }
}
