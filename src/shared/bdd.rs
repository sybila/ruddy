use std::{
    cell::Cell,
    rc::{Rc, Weak},
};

use crate::node_id::NodeId;

#[derive(Debug, Clone)]
pub struct Bdd {
    pub(crate) root: Rc<Cell<NodeId>>,
}

impl Bdd {
    pub(crate) fn new(root: NodeId) -> Self {
        Self {
            root: Rc::new(Cell::new(root)),
        }
    }

    pub(crate) fn root_weak(&self) -> Weak<Cell<NodeId>> {
        Rc::downgrade(&self.root)
    }

    pub fn is_true(&self) -> bool {
        self.root.get().is_one()
    }

    pub fn is_false(&self) -> bool {
        self.root.get().is_zero()
    }
}

impl PartialEq for Bdd {
    fn eq(&self, other: &Self) -> bool {
        self.root.get() == other.root.get()
    }
}

impl Eq for Bdd {}
