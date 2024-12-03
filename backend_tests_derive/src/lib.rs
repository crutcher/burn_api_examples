use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn generate_backend_tests(
    _attr: TokenStream,
    item: TokenStream,
) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_name = input.sig.ident.clone();
    let fn_name_str = fn_name.to_string();

    // Define the test generators
    let generated_tests = vec![
        ("wgpu", quote!(burn::backend::Wgpu)),
        ("ndarray", quote!(burn::backend::NdArray)),
    ];

    // Generate the test functions
    let test_fns = generated_tests.into_iter().map(|(name, backend)| {
        let test_fn_name = syn::Ident::new(&format!("{}_{}", fn_name_str, name), fn_name.span());
        quote! {
            #[test]
            fn #test_fn_name() {
                #fn_name::<#backend>(Default::default());
            }
        }
    });

    // Combine the original function and the generated test functions
    let result = quote! {
        #input
        #(#test_fns)*
    };

    result.into()
}
