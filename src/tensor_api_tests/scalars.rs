#[cfg(test)]
mod test {
    use backend_tests_derive::generate_backend_tests;
    use burn::tensor::{
        backend::Backend,
        cast::ToElement,
        Tensor,
    };

    #[generate_backend_tests]
    fn test_into_scalar<B: Backend>(device: B::Device) {
        let tensor: Tensor<B, 2> = Tensor::ones([1, 1], &device);

        let scalar = tensor.clone().into_scalar();
        assert_eq!(scalar.to_f32(), 1.0);
    }
}
