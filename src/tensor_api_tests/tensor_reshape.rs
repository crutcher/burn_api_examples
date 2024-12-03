#[cfg(test)]
mod tests {
    use crate::tensor_assertions::*;

    use backend_tests_derive::generate_backend_tests;
    use burn::tensor::{
        backend::Backend,
        Shape,
        Tensor,
    };

    #[generate_backend_tests]
    fn test_reshape<B: Backend>(device: B::Device) {
        let tensor: Tensor<B, 3> = Tensor::from_floats(
            [
                [[1., 2., 3.], [4., 5., 6.]],
                [[7., 8., 9.], [10., 11., 12.]],
            ],
            &device,
        );
        assert_eq!(tensor.shape(), Shape::new([2, 2, 3]));

        assert_tensor_eq_floats!(
            &tensor.clone().reshape([2, -1]),
            [[1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.]]
        );

        assert_tensor_eq_floats!(
            &tensor.clone().reshape([4, 3]),
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]]
        );
    }

    #[generate_backend_tests]
    fn test_flatten<B: Backend>(device: B::Device) {
        let tensor: Tensor<B, 3> = Tensor::from_floats(
            [
                [[1., 2., 3.], [4., 5., 6.]],
                [[7., 8., 9.], [10., 11., 12.]],
            ],
            &device,
        );
        assert_eq!(tensor.shape(), Shape::new([2, 2, 3]));

        let f: Tensor<B, 2> = tensor.flatten(0, 1);

        assert_tensor_eq_floats!(
            &f,
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]]
        );
    }
}
