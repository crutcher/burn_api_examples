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
    fn test_transpose<B: Backend>(device: B::Device) {
        let tensor: Tensor<B, 2> = Tensor::from_floats([[1., 2.], [3., 4.]], &device);
        assert_eq!(tensor.shape(), Shape::new([2, 2]));

        let tensor = tensor.transpose();
        assert_eq!(tensor.shape(), Shape::new([2, 2]));
        assert_tensor_eq_floats!(&tensor, [[1., 3.], [2., 4.]]);
    }

    #[generate_backend_tests]
    fn test_swap_dims<B: Backend>(device: B::Device) {
        let tensor: Tensor<B, 3> = Tensor::from_floats(
            [
                [[1., 2., 3.], [4., 5., 6.]],
                [[7., 8., 9.], [10., 11., 12.]],
            ],
            &device,
        );
        assert_eq!(tensor.shape(), Shape::new([2, 2, 3]));

        let tensor = tensor.swap_dims(1, 2);

        assert_eq!(tensor.shape(), Shape::new([2, 3, 2]));
        assert_tensor_eq_floats!(
            &tensor,
            [
                [[1., 4.], [2., 5.], [3., 6.]],
                [[7., 10.], [8., 11.], [9., 12.]],
            ]
        );
    }
}
