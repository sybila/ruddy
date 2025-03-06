/// Performs value-to-value conversions while consuming the input, without any checks.
///
/// This trait is intended for cases where the conversion is **expected** to *succeed*
/// and be *lossless*. As such, it inherently less safe than [`TryFrom`], but typically faster.
///
/// One should always prefer implementing `UncheckedFrom` over [`UncheckedInto`] because
/// implementing `UncheckedFrom` automatically provides one with an implementation
/// of [`UncheckedInto`] thanks to the blanket implementation in the library.
///
/// Prefer using [`UncheckedInto`] over using `UncheckedFrom` when specifying
/// trait bounds on generic functions. This way, types that directly implement
/// [`UncheckedInto`] can be used as arguments as well.
///
/// # Safety
///
/// Using `UncheckedFrom` for lossy conversions can lead to subtle bugs,
/// undefined behavior, or panics. It is recommended to at least include
/// `debug_assert!` checks to ensure validity.
pub trait UncheckedFrom<T>: Sized {
    /// Converts to this type from the input type.
    fn unchecked_from(value: T) -> Self;
}

/// A value-to-value conversion that consumes the input value, without any checks.
/// The opposite of [`UncheckedFrom`].
///
/// This trait is intended for cases where the conversion is **expected** to *succeed*
/// and be *lossless*. As such, it inherently less safe than [`TryInto`], but typically faster.
///
/// One should avoid implementing `UncheckedInto` and implement [`UncheckedFrom`] instead.
/// Implementing [`UncheckedFrom`] automatically provides one with an implementation of
/// `UncheckedInto` thanks to the blanket implementation in the library.

/// Prefer using `UncheckedInto` over [`UncheckedFrom`] when specifying trait bounds
/// on a generic function to ensure that types that only implement `UncheckedInto`
///  can be used as well.
///
/// # Safety
///
/// Using `UncheckedInto` for lossy conversions can lead to subtle bugs,
/// undefined behavior, or panics. It is recommended to at least include
/// `debug_assert!` checks to ensure validity.
pub trait UncheckedInto<T>: Sized {
    /// Converts this type into the (usually inferred) input type.
    fn unchecked_into(self) -> T;
}

// `UncheckedFrom` implies `UncheckedInto`.
impl<T, U> UncheckedInto<U> for T
where
    U: UncheckedFrom<T>,
{
    fn unchecked_into(self) -> U {
        U::unchecked_from(self)
    }
}

// `UncheckedFrom` (and thus `UncheckedInto`) is reflexive.
impl<T> UncheckedFrom<T> for T {
    fn unchecked_from(value: T) -> Self {
        value
    }
}

// A macro that can derive `UncheckedFrom` for types where `From` or `as` already exists.
// The reason why this isn't a blanket implementation for all applicable types is that
// the blanket implementation could break if upstream crates later add extra implementations
// of `From` for types that don't have it yet.
macro_rules! derive_unchecked_from {
    ($A:ident => $B:ident) => {
        impl UncheckedFrom<$A> for $B {
            fn unchecked_from(x: $A) -> Self {
                x.into()
            }
        }
    };
    ($A:ident => $($B:ident),+) => {
        $(derive_unchecked_from!($A => $B);)+
    };
    ($A:ident as $B:ident) => {
        impl UncheckedFrom<$A> for $B {
            fn unchecked_from(x: $A) -> Self {
                debug_assert!($B::try_from(x).is_ok());
                x as $B
            }
        }
    };
    ($A:ident as $($B:ident),+) => {
        $(derive_unchecked_from!($A as $B);)+
    };
}

derive_unchecked_from!(u16 => u32, u64, u128, usize);
derive_unchecked_from!(u32 => u64, u128);
derive_unchecked_from!(u64 => u128);
derive_unchecked_from!(u64 as u16, u32, usize);
derive_unchecked_from!(u32 as u16, usize);
