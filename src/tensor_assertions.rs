use burn::tensor::{
    backend::Backend,
    Tensor,
    TensorData,
};

pub fn _assert_tensor_eq<B: Backend, const AN: usize, const BN: usize>(
    actual: &Tensor<B, AN>,
    expected: &Tensor<B, BN>,
    strict: bool,
) {
    let a_data = actual.clone().to_data();
    let e_data = expected.clone().to_data();
    a_data.assert_eq(&e_data, strict);
}

pub fn _assert_tensor_eq_floats<B: Backend, const AN: usize, F: Into<TensorData>>(
    actual: &Tensor<B, AN>,
    expected: F,
    strict: bool,
) {
    let expected_tensor: Tensor<B, AN> = Tensor::from_floats(expected, &actual.device());
    _assert_tensor_eq(actual, &expected_tensor, strict);
}

#[macro_export]
macro_rules! assert_tensor_eq {
    ($actual: expr, $expected: expr) => {
        _assert_tensor_eq($actual, $expected, false);
    };
    ($actual: expr, $expected: expr, strict: bool) => {
        _assert_tensor_eq($actual, $expected, strict);
    };
}

#[macro_export]
macro_rules! assert_tensor_eq_floats {
    ($actual: expr, $expected: expr) => {
        _assert_tensor_eq_floats($actual, $expected, false);
    };
    ($actual: expr, $expected: expr, $strict: expr) => {
        _assert_tensor_eq_floats($actual, $expected, $strict);
    };
}

/*
TODO(crutcher): work out correct way to export/import macros.
*/
#[allow(unused_imports)]
pub(crate) use assert_tensor_eq;
pub(crate) use assert_tensor_eq_floats;
