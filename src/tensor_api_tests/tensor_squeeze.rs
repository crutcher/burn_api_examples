#[cfg(test)]
mod tests {
    use crate::tensor_assertions::*;
    use burn::prelude::Shape;

    use backend_tests_derive::generate_backend_tests;
    use burn::tensor::{
        backend::Backend,
        Tensor,
    };

    #[generate_backend_tests]
    fn test_unsqueeze<B: Backend>(device: B::Device) {
        let tensor: Tensor<B, 3> = Tensor::from_floats(
            [
                [[1., 2., 3.], [4., 5., 6.]],
                [[7., 8., 9.], [10., 11., 12.]],
            ],
            &device,
        );
        assert_eq!(tensor.shape(), Shape::new([2, 2, 3]));

        let tensor: Tensor<B, 5> = tensor.unsqueeze();
        assert_eq!(tensor.shape(), Shape::new([1, 1, 2, 2, 3]));

        assert_tensor_eq_floats!(
            &tensor,
            [[[
                [[1., 2., 3.], [4., 5., 6.]],
                [[7., 8., 9.], [10., 11., 12.]],
            ]]]
        );
    }

    #[generate_backend_tests]
    fn test_unsqueeze_dim<B: Backend>(device: B::Device) {
        let tensor: Tensor<B, 3> = Tensor::from_floats(
            [
                [[1., 2., 3.], [4., 5., 6.]],
                [[7., 8., 9.], [10., 11., 12.]],
            ],
            &device,
        );
        assert_eq!(tensor.shape(), Shape::new([2, 2, 3]));

        let tensor: Tensor<B, 4> = tensor.unsqueeze_dim(1);
        assert_eq!(tensor.shape(), Shape::new([2, 1, 2, 3]));
        assert_tensor_eq_floats!(
            &tensor,
            [
                [[[1., 2., 3.], [4., 5., 6.]]],
                [[[7., 8., 9.], [10., 11., 12.]]],
            ]
        );
    }

    #[generate_backend_tests]
    fn test_unsqueeze_dims<B: Backend>(device: B::Device) {
        let tensor: Tensor<B, 3> = Tensor::from_floats(
            [
                [[1., 2., 3.], [4., 5., 6.]],
                [[7., 8., 9.], [10., 11., 12.]],
            ],
            &device,
        );
        assert_eq!(tensor.shape(), Shape::new([2, 2, 3]));

        let a: Tensor<B, 5> = tensor.clone().unsqueeze_dims(&[0, -1]);
        assert_eq!(a.shape(), Shape::new([1, 2, 2, 3, 1]));

        let b: Tensor<B, 4> = tensor.clone().unsqueeze_dims(&[-2]);
        assert_eq!(b.shape(), Shape::new([2, 2, 1, 3]));
    }

    #[generate_backend_tests]
    fn test_unsqueeze_dims_bug<B: Backend>(device: B::Device) {
        let tensor: Tensor<B, 3> = Tensor::from_floats(
            [
                [[1., 2., 3.], [4., 5., 6.]],
                [[7., 8., 9.], [10., 11., 12.]],
            ],
            &device,
        );
        assert_eq!(tensor.shape(), Shape::new([2, 2, 3]));

        // FIXME(crutcher): this panics, it appears to be a bug in burn.
        let tensor: Tensor<B, 5> = tensor.unsqueeze_dims(&[0, -2]);
        assert_eq!(tensor.shape(), Shape::new([1, 2, 2, 1, 3]));
    }
}
