use smallvec::SmallVec;
use crate::core::error::CoreError;
use crate::core::error::Result;

/*
 Defines the shape for every tensor op. Can go from 2D to 4D
 */

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(SmallVec<[usize; 4]>);

impl Shape {
    pub fn new(dims: &[usize]) -> Self{
        Self(SmallVec::from_slice(dims))
    }

    pub fn scalar(&self) -> Self{
        Self(SmallVec::new())
    }

    //Tells you how many dimensions there are
    pub fn rank(&self) -> usize{
        self.0.len()
    }


    //Tells you what those dimensions are
    pub fn dims(&self) -> &[usize]  {
        &self.0
    }

    pub fn dim(&self, i: usize) -> Result<usize>{
        self.0.get(i)
            .copied() // Convert &usize to usize
            .ok_or_else(|| CoreError::OutOfBounds {
                op: "shape_access",
                index: i,
                size: self.rank(),
        })
    }

    pub fn numel(&self) -> usize {
         
    }

    pub fn is_scalar(&self) -> bool{

    }

    pub fn is_vector(&self) -> bool{

    }

    pub fn is_matrix(&self) -> bool{

    }

    // validate matmul compatibility — returns Err if incompatible
    pub fn check_matmul(&self, rhs: &Shape) -> crate::error::Result<Shape>

    // validate elementwise op compatibility
    pub fn check_elementwise(&self, rhs: &Shape) -> crate::error::Result<()>
}
