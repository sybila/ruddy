use std::fmt::Display;
use std::io::{self, ErrorKind, Read, Write};

use super::Bdd;
use super::bdd::{Bdd16, Bdd32, Bdd64, BddAny, BddImpl, BddInner};
use crate::DeserializeIdError;
use crate::bdd_node::BddNodeAny;
use crate::{node_id::NodeIdAny, variable_id::VarIdPackedAny};

/// An error than can occur while deserializing a BDD.
#[derive(Debug)]
#[non_exhaustive]
pub enum BddDeserializationError {
    /// The low child id is invalid.
    InvalidLowChild(DeserializeIdError),
    /// The high child id is invalid.
    InvalidHighChild(DeserializeIdError),
    /// The variable id is invalid.
    InvalidVariable(DeserializeIdError),
    /// The root id of the BDD is invalid.
    InvalidRoot(DeserializeIdError),
    /// The width of the BDD is invalid.
    InvalidWidth(u8),
    /// The expected delimiter '|' was not found.
    MissingDelimiter,
    /// The BDD contains no nodes.
    EmptyBdd,
    /// Any other IO error.
    IoError(io::Error),
}

impl Display for BddDeserializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BddDeserializationError::InvalidLowChild(err) => {
                write!(f, "Low child id: {err}")
            }
            BddDeserializationError::InvalidHighChild(err) => {
                write!(f, "High child id: {err}")
            }
            BddDeserializationError::InvalidVariable(err) => {
                write!(f, "Variable id: {err}")
            }
            BddDeserializationError::InvalidRoot(err) => write!(f, "Invalid root id: {err}"),
            BddDeserializationError::InvalidWidth(width) => {
                write!(f, "Invalid width: {width}. Expected 16, 32 or 64.")
            }
            BddDeserializationError::MissingDelimiter => {
                write!(f, "Missing delimiter in serialized string.")
            }
            BddDeserializationError::EmptyBdd => write!(f, "The BDD does not contain any nodes."),
            BddDeserializationError::IoError(err) => write!(f, "IO error: {err}"),
        }
    }
}

impl From<io::Error> for BddDeserializationError {
    fn from(err: io::Error) -> Self {
        BddDeserializationError::IoError(err)
    }
}

type NodeId<T> = <T as BddAny>::Id;
type VarId<T> = <T as BddAny>::VarId;
type Node<T> = <T as BddAny>::Node;

// Unfortunately, we have to use a macro here.
// It is not possible to specify an associated `BYTE_WIDTH` constant inside
// `NodeIdAny` and have a function `to_le_bytes -> [u8; BYTE_WIDTH]` until
// https://github.com/rust-lang/rust/issues/132980
macro_rules! impl_byte_conversions {
    ($($Bdd:ident),*) => {
        $(
        impl $Bdd {
            /// Serializes the BDD to the `output` byte stream.
            pub fn write_as_bytes(&self, output: &mut dyn Write) -> io::Result<()> {
                output.write_all(&self.root.to_le_bytes())?;

                for node in &self.nodes {
                    output.write_all(&node.variable.to_le_bytes())?;
                    output.write_all(&node.low.to_le_bytes())?;
                    output.write_all(&node.high.to_le_bytes())?;
                }

                Ok(())
            }

            /// Deserializes a BDD from the `input` byte stream.
            pub fn read_as_bytes(input: &mut dyn Read) -> Result<Self, BddDeserializationError> {
                const WIDTH: usize = std::mem::size_of::<VarId<$Bdd>>();
                let mut buffer = [0u8; 3 * WIDTH];
                let mut nodes = Vec::new();

                // Read the root first
                let mut root_buffer = [0u8; WIDTH];
                input.read_exact(&mut root_buffer)?;
                let root = <NodeId<$Bdd>>::from_le_bytes(root_buffer).map_err(BddDeserializationError::InvalidRoot)?;

                loop {
                    match input.read_exact(&mut buffer) {
                        Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                            if nodes.is_empty() {
                                return Err(BddDeserializationError::EmptyBdd);
                            }

                            return unsafe {
                                Ok(Self::new_unchecked(
                                    root,
                                    nodes,
                                ))
                            }
                        }
                        _ => (),
                    }

                    let var_slice: [u8; WIDTH] = buffer[0..WIDTH].try_into().unwrap();
                    let low_slice: [u8; WIDTH] = buffer[WIDTH..2 * WIDTH].try_into().unwrap();
                    let high_slice: [u8; WIDTH] = buffer[2 * WIDTH..3 * WIDTH].try_into().unwrap();

                    let variable = <VarId<Self>>::from_le_bytes(var_slice);
                    let low = <NodeId<Self>>::from_le_bytes(low_slice).map_err(BddDeserializationError::InvalidLowChild)?;
                    let high = <NodeId<Self>>::from_le_bytes(high_slice).map_err(BddDeserializationError::InvalidHighChild)?;

                    let node = unsafe { <Node<Self>>::new_unchecked(variable, low, high) };

                    if !node.is_terminal() && variable.is_undefined() {
                        return Err(BddDeserializationError::InvalidVariable(
                            DeserializeIdError::InvalidId,
                        ));
                    }

                    nodes.push(node);
                }
            }

            #[allow(dead_code)]
            /// Serializes the BDD to a byte vector.
            pub fn to_bytes(&self) -> Vec<u8> {
                let mut bytes = Vec::new();
                self.write_as_bytes(&mut bytes).unwrap();
                bytes
            }

            #[allow(dead_code)]
            /// Deserializes a BDD from a byte slice.
            pub fn from_bytes(bytes: &mut &[u8]) -> Result<Self, BddDeserializationError> {
                $Bdd::read_as_bytes(bytes)
            }
        }
    )*
    };
}

impl_byte_conversions!(Bdd16, Bdd32, Bdd64);

impl<TNodeId: NodeIdAny, TVarId: VarIdPackedAny> BddImpl<TNodeId, TVarId> {
    /// Serializes the BDD as a string to the `output` stream.
    pub fn write_as_serialized_string(&self, output: &mut dyn Write) -> io::Result<()> {
        write!(output, "{}|", self.root())?;
        for node in &self.nodes {
            // write the variable using debug formatting to ensure it retains
            // the packed information, and we don't have to recompute it
            write!(output, "{},", node.variable().to_string_packed())?;
            write!(output, "{},", node.low())?;
            write!(output, "{}|", node.high())?;
        }

        Ok(())
    }

    /// Deserializes a BDD from a string.
    pub fn from_serialized_string(buffer: &str) -> Result<Self, BddDeserializationError> {
        let mut nodes = Vec::new();

        let (root_str, rest) = buffer
            .split_once('|')
            .ok_or(BddDeserializationError::MissingDelimiter)?;
        let root = root_str
            .parse::<TNodeId>()
            .map_err(BddDeserializationError::InvalidRoot)?;

        for node_str in rest.split('|').filter(|s| !s.is_empty()) {
            let mut node_parts = node_str.split(',');

            let variable = TVarId::from_string_packed(node_parts.next().unwrap_or_default())
                .map_err(BddDeserializationError::InvalidVariable)?;

            let low = node_parts
                .next()
                .ok_or(BddDeserializationError::MissingDelimiter)?
                .parse::<TNodeId>()
                .map_err(BddDeserializationError::InvalidLowChild)?;

            let high = node_parts
                .next()
                .ok_or(BddDeserializationError::MissingDelimiter)?
                .parse::<TNodeId>()
                .map_err(BddDeserializationError::InvalidHighChild)?;

            let node = unsafe { <Node<Self>>::new_unchecked(variable, low, high) };

            if !node.is_terminal() && variable.is_undefined() {
                return Err(BddDeserializationError::InvalidVariable(
                    DeserializeIdError::InvalidId,
                ));
            }

            nodes.push(node);
        }

        if nodes.is_empty() {
            return Err(BddDeserializationError::EmptyBdd);
        }

        unsafe { Ok(Self::new_unchecked(root, nodes)) }
    }

    /// Deserializes a BDD from a string from the `input` stream.
    #[allow(dead_code)]
    pub fn read_as_serialized_string(
        input: &mut dyn Read,
    ) -> Result<Self, BddDeserializationError> {
        let mut buffer = String::new();
        input.read_to_string(&mut buffer)?;
        buffer.retain(|c| !c.is_whitespace());
        Self::from_serialized_string(&buffer)
    }

    /// Serializes the BDD to a string.
    #[allow(dead_code)]
    pub fn to_serialized_string(&self) -> String {
        let mut bytes = Vec::new();
        self.write_as_serialized_string(&mut bytes).unwrap();
        String::from_utf8(bytes).unwrap()
    }
}

// }

impl Bdd {
    /// Serializes the `Bdd` to a string to the `output` stream.
    pub fn write_as_serialized_string(&self, output: &mut dyn Write) -> io::Result<()> {
        match &self.0 {
            BddInner::Size16(bdd) => {
                write!(output, "16|")?;
                bdd.write_as_serialized_string(output)
            }
            BddInner::Size32(bdd) => {
                write!(output, "32|")?;
                bdd.write_as_serialized_string(output)
            }
            BddInner::Size64(bdd) => {
                write!(output, "64|")?;
                bdd.write_as_serialized_string(output)
            }
        }
    }

    /// Serializes the `Bdd` to a `String`.
    pub fn to_serialized_string(&self) -> String {
        let mut bytes = Vec::new();
        self.write_as_serialized_string(&mut bytes).unwrap();
        String::from_utf8(bytes).unwrap()
    }

    /// Deserializes a `Bdd` from a string.
    pub fn from_serialized_string(buffer: &str) -> Result<Self, BddDeserializationError> {
        let (width, rest) = buffer
            .split_once('|')
            .ok_or(io::Error::new(ErrorKind::InvalidData, "invalid width"))?;

        let width = width
            .parse::<u8>()
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "invalid width"))?;

        match width {
            16 => Ok(Bdd16::from_serialized_string(rest)?.into()),
            32 => Ok(Bdd32::from_serialized_string(rest)?.into()),
            64 => Ok(Bdd64::from_serialized_string(rest)?.into()),
            _ => Err(BddDeserializationError::InvalidWidth(width)),
        }
    }

    /// Deserializes a `Bdd` from a string from the `input` stream.
    pub fn read_as_serialized_string(
        input: &mut dyn Read,
    ) -> Result<Self, BddDeserializationError> {
        let mut buffer = String::new();
        input.read_to_string(&mut buffer)?;
        buffer.retain(|c| !c.is_whitespace());
        Self::from_serialized_string(&buffer)
    }

    /// Serializes a `Bdd` to the `output` byte stream.
    pub fn write_as_bytes(&self, output: &mut dyn Write) -> io::Result<()> {
        match &self.0 {
            BddInner::Size16(bdd) => {
                output.write_all(&[16])?;
                bdd.write_as_bytes(output)
            }
            BddInner::Size32(bdd) => {
                output.write_all(&[32])?;
                bdd.write_as_bytes(output)
            }
            BddInner::Size64(bdd) => {
                output.write_all(&[64])?;
                bdd.write_as_bytes(output)
            }
        }
    }

    /// Deserializes a `Bdd` from the `input` byte stream.
    pub fn read_as_bytes(input: &mut dyn Read) -> Result<Bdd, BddDeserializationError> {
        let mut width = [0u8; 1];
        input.read_exact(&mut width)?;

        match width[0] {
            16 => Ok(Bdd16::read_as_bytes(input)?.into()),
            32 => Ok(Bdd32::read_as_bytes(input)?.into()),
            64 => Ok(Bdd64::read_as_bytes(input)?.into()),
            _ => Err(BddDeserializationError::InvalidWidth(width[0])),
        }
    }

    /// Serializes the `Bdd` to a byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.write_as_bytes(&mut bytes).unwrap();
        bytes
    }

    /// Deserializes a `Bdd` from a byte slice.
    pub fn from_bytes(bytes: &mut &[u8]) -> Result<Self, BddDeserializationError> {
        let mut input = bytes;
        Bdd::read_as_bytes(&mut input)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        split::{
            Bdd,
            bdd::{Bdd16, Bdd32, Bdd64, BddAny},
        },
        variable_id::VariableId,
    };

    fn ripple_carry_adder(num_vars: u64, var_offset: u64) -> Bdd {
        let mut result = Bdd::new_false();
        for x in 0..(num_vars / 2) {
            let x1 = Bdd::new_literal(VariableId::new_long(x + var_offset).unwrap(), true);
            let x2 = Bdd::new_literal(
                VariableId::new_long(x + num_vars / 2 + var_offset).unwrap(),
                true,
            );
            result = result.or(&x1.and(&x2));
        }
        result
    }

    fn test_bdd_byte_conversion(bdd: Bdd) {
        let bytes = bdd.to_bytes();
        let mut slice: &[u8] = &bytes;
        let bdd_from_bytes = Bdd::from_bytes(&mut slice).unwrap();
        assert!(bdd.structural_eq(&bdd_from_bytes));
    }

    macro_rules! test_constant_bdd_byte_conversion {
        ($bdd:ident, $func:ident) => {
            let bdd = $bdd::$func();
            let bytes = bdd.to_bytes();
            let mut slice: &[u8] = &bytes;
            let bdd_from_bytes = $bdd::from_bytes(&mut slice).unwrap();
            assert!(bdd.structural_eq(&bdd_from_bytes));
        };
    }

    #[test]
    fn bdd16_constant_byte_conversions() {
        test_constant_bdd_byte_conversion!(Bdd16, new_true);
        test_constant_bdd_byte_conversion!(Bdd16, new_false);
    }

    #[test]
    fn bdd32_constant_byte_conversions() {
        test_constant_bdd_byte_conversion!(Bdd32, new_true);
        test_constant_bdd_byte_conversion!(Bdd32, new_false);
    }

    #[test]
    fn bdd64_constant_byte_conversions() {
        test_constant_bdd_byte_conversion!(Bdd64, new_true);
        test_constant_bdd_byte_conversion!(Bdd64, new_false);
    }

    #[test]
    fn bdd16_byte_conversions() {
        test_bdd_byte_conversion(ripple_carry_adder(16, 0));
    }

    #[test]
    fn bdd32_byte_conversions() {
        test_bdd_byte_conversion(ripple_carry_adder(16, u16::MAX as u64));
    }

    #[test]
    fn bdd64_byte_conversions() {
        test_bdd_byte_conversion(ripple_carry_adder(16, u32::MAX as u64));
    }

    fn test_bdd_string_conversion(bdd: Bdd) {
        let s = bdd.to_serialized_string();
        println!("Serialized string: {s}");
        let bdd_from_s = Bdd::from_serialized_string(&s).unwrap();
        assert!(bdd.structural_eq(&bdd_from_s));
    }

    macro_rules! test_constant_bdd_string_conversion {
        ($bdd:ident, $func:ident) => {
            let bdd = $bdd::$func();
            let s = bdd.to_serialized_string();
            let bdd_from_s = $bdd::from_serialized_string(&s).unwrap();
            assert!(bdd.structural_eq(&bdd_from_s));
        };
    }

    #[test]
    fn bdd16_constant_string_conversions() {
        test_constant_bdd_string_conversion!(Bdd16, new_true);
        test_constant_bdd_string_conversion!(Bdd16, new_false);
    }

    #[test]
    fn bdd32_constant_string_conversions() {
        test_constant_bdd_string_conversion!(Bdd32, new_true);
        test_constant_bdd_string_conversion!(Bdd32, new_false);
    }

    #[test]
    fn bdd64_constant_string_conversions() {
        test_constant_bdd_string_conversion!(Bdd64, new_true);
        test_constant_bdd_string_conversion!(Bdd64, new_false);
    }

    #[test]
    fn bdd16_string_conversions() {
        test_bdd_string_conversion(ripple_carry_adder(16, 0));
    }

    #[test]
    fn bdd32_string_conversions() {
        test_bdd_string_conversion(ripple_carry_adder(16, u16::MAX as u64));
    }

    #[test]
    fn bdd64_string_conversions() {
        test_bdd_string_conversion(ripple_carry_adder(16, u32::MAX as u64));
    }
}
