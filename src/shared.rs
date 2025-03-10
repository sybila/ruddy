use std::{
    cell::Cell,
    rc::{Rc, Weak},
};

use crate::{
    conversion::UncheckedInto, node_id::NodeId, node_table::NodeTable, variable_id::VariableId,
};

use replace_with::replace_with_or_default;

#[derive(Debug, Default)]
pub struct BddManager {
    unique_table: NodeTable,
    roots: Vec<Weak<Cell<NodeId>>>,
}

#[derive(Debug)]
pub struct Bdd {
    #[allow(dead_code)]
    root: Rc<Cell<NodeId>>,
}

impl BddManager {
    pub fn new() -> Self {
        Self {
            unique_table: NodeTable::Size16(Default::default()),
            roots: Default::default(),
        }
    }

    pub fn new_bdd_false(&self) -> Bdd {
        Bdd {
            root: Rc::new(Cell::new(NodeId::zero())),
        }
    }

    pub fn new_bdd_true(&self) -> Bdd {
        Bdd {
            root: Rc::new(Cell::new(NodeId::one())),
        }
    }

    fn grow(&mut self) {
        replace_with_or_default(&mut self.unique_table, |table| match table {
            NodeTable::Size16(table) => NodeTable::Size32(table.into()),
            NodeTable::Size32(table) => NodeTable::Size64(table.into()),
            table64 => table64,
        });
    }

    fn grow_to_64(&mut self) {
        replace_with_or_default(&mut self.unique_table, |table| match table {
            NodeTable::Size16(table) => NodeTable::Size64(table.into()),
            NodeTable::Size32(table) => NodeTable::Size64(table.into()),
            table64 => table64,
        });
    }

    pub fn new_bdd_literal(&mut self, variable: VariableId, value: bool) -> Bdd {
        match &self.unique_table {
            NodeTable::Size16(_) if variable.fits_only_in_packed64() => {
                self.grow_to_64();
            }
            NodeTable::Size32(_) if variable.fits_only_in_packed64() => {
                self.grow();
            }
            NodeTable::Size16(_) if variable.fits_only_in_packed32() => {
                self.grow();
            }
            _ => {}
        }

        if self.unique_table.is_full() {
            self.grow();
        }

        let root = match &mut self.unique_table {
            NodeTable::Size16(table) => {
                let root_id = table
                    .ensure_literal(variable.unchecked_into(), value)
                    .expect("ensuring literal after growth should always succeed");
                Rc::new(Cell::new(root_id.into()))
            }
            NodeTable::Size32(table) => {
                let root_id = table
                    .ensure_literal(variable.unchecked_into(), value)
                    .expect("ensuring literal after growth should always succeed");
                Rc::new(Cell::new(root_id.into()))
            }
            NodeTable::Size64(table) => {
                let root_id = table
                    .ensure_literal(variable.unchecked_into(), value)
                    .expect("TODO: 64-bit ensure_literal failed");
                Rc::new(Cell::new(root_id.into()))
            }
        };

        self.roots.push(Rc::downgrade(&root));
        Bdd { root }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        node_id::{NodeId32, NodeId64, NodeIdAny},
        variable_id::{VarIdPacked32, VarIdPacked64, VariableId},
    };

    #[test]
    fn manager_grows_from_16_to_32() {
        let mut manager = BddManager::new();
        let nodes = 1 << 16;
        for i in 0..nodes {
            let v = VariableId::new(i);
            manager.new_bdd_literal(v, true);
        }
        assert!(matches!(manager.unique_table, NodeTable::Size32(_)));
        assert_eq!(manager.unique_table.node_count(), nodes as usize + 2);

        match &manager.unique_table {
            NodeTable::Size32(table) => {
                for i in 2..nodes {
                    let node = unsafe { table.get_node_unchecked(NodeId32::new(i)) };
                    assert_eq!(node.low, NodeId32::zero());
                    assert_eq!(node.high, NodeId32::one());
                    assert_eq!(node.variable, VarIdPacked32::new(i - 2));
                }
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn adding_32_bit_variable_to_16_bit_manager_grows_to_32_bit() {
        let mut manager = BddManager::new();
        let var_num = u32::from(u16::MAX);
        let variable = VariableId::new(var_num);
        manager.new_bdd_literal(variable, true);

        assert!(matches!(manager.unique_table, NodeTable::Size32(_)));
        assert_eq!(manager.unique_table.node_count(), 3);

        match &manager.unique_table {
            NodeTable::Size32(table) => {
                let node = unsafe { table.get_node_unchecked(NodeId32::new(2)) };
                assert_eq!(node.low, NodeId32::zero());
                assert_eq!(node.high, NodeId32::one());
                assert_eq!(node.variable, VarIdPacked32::new(var_num));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn adding_64_bit_variable_to_16_bit_manager_grows_to_64_bit() {
        let mut manager = BddManager::new();
        let var_num = u64::from(u32::MAX);
        let variable = VariableId::new_long(var_num).unwrap();
        manager.new_bdd_literal(variable, true);

        assert!(matches!(manager.unique_table, NodeTable::Size64(_)));
        assert_eq!(manager.unique_table.node_count(), 3);

        match &manager.unique_table {
            NodeTable::Size64(table) => {
                let node = unsafe { table.get_node_unchecked(NodeId64::new(2)) };
                assert_eq!(node.low, NodeId64::zero());
                assert_eq!(node.high, NodeId64::one());
                assert_eq!(node.variable, VarIdPacked64::new(var_num));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn adding_64_bit_variable_to_32_bit_manager_grows_to_64_bit() {
        let mut manager = BddManager::new();
        manager.grow();
        assert!(matches!(manager.unique_table, NodeTable::Size32(_)));
        assert_eq!(manager.unique_table.node_count(), 2);

        let var_num = u64::from(u32::MAX);
        let variable = VariableId::new_long(var_num).unwrap();
        manager.new_bdd_literal(variable, true);

        assert!(matches!(manager.unique_table, NodeTable::Size64(_)));
        assert_eq!(manager.unique_table.node_count(), 3);

        match &manager.unique_table {
            NodeTable::Size64(table) => {
                let node = unsafe { table.get_node_unchecked(NodeId64::new(2)) };
                assert_eq!(node.low, NodeId64::zero());
                assert_eq!(node.high, NodeId64::one());
                assert_eq!(node.variable, VarIdPacked64::new(var_num));
            }
            _ => unreachable!(),
        }
    }
}
