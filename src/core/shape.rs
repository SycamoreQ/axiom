use crate::core::error::CoreError;
use crate::core::error::Result;
use smallvec::SmallVec;

/*
Defines the shape for every tensor op. Can go from 2D to 4D
*/

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(SmallVec<[usize; 4]>);

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        Self(SmallVec::from_slice(dims))
    }

    pub fn scalar(&self) -> Self {
        Self(SmallVec::new())
    }

    //Tells you how many dimensions there are
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    //Tells you what those dimensions are
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn dim(&self, i: usize) -> Result<usize> {
        self.0
            .get(i)
            .copied() // Convert &usize to usize
            .ok_or_else(|| CoreError::OutOfBounds {
                op: "shape_access",
                index: i,
                size: self.rank(),
            })
    }

    pub fn numel(&self) -> usize {
        self.0.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.rank() == 0
    }

    pub fn is_vector(&self) -> bool {
        self.rank() == 1
    }

    pub fn is_matrix(&self) -> bool {
        self.rank() == 2
    }

    pub fn is_tensor(&self) -> bool {
        self.rank() >= 3
    }

    pub fn rows(&self) -> Result<usize> {
        if self.is_matrix() || self.rank() > 2 {
            // For a matrix [Rows, Cols], Rows is at index 0
            Ok(self.0[0])
        } else {
            Err(CoreError::RankMismatch {
                op: "rows",
                expected: 2,
                got: self.rank(),
            })
        }
    }

    pub fn cols(&self) -> Result<usize> {
        if self.is_matrix() || self.rank() > 2 {
            // For a matrix [Rows, Cols], Cols is at index 1
            Ok(self.0[1])
        } else {
            Err(CoreError::RankMismatch {
                op: "cols",
                expected: 2,
                got: self.rank(),
            })
        }
    }

    // validate matmul compatibility — returns Err if incompatible
    pub fn matmul_check(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
        //Ensure both are at least Rank 2
        if lhs.rank() < 2 || rhs.rank() < 2 {
            return Err(CoreError::RankMismatch {
                op: "matmul",
                expected: 2,
                got: std::cmp::min(lhs.rank(), rhs.rank()),
            });
        }

        // lhs is [..., M, K], rhs is [..., K, N]
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();

        let m = lhs_dims[lhs.rank() - 2];
        let k_lhs = lhs_dims[lhs.rank() - 1];
        let k_rhs = rhs_dims[rhs.rank() - 2];
        let n = rhs_dims[rhs.rank() - 1];

        //The "K" dimensions must match
        if k_lhs != k_rhs {
            return Err(CoreError::ShapeMismatch {
                op: "matmul",
                lhs: lhs.dims().to_vec(),
                rhs: rhs.dims().to_vec(),
            });
        }

        Ok(Shape::new(&[m, n]))
    }

    // validate elementwise op compatibility
    pub fn elementwise_check(lhs: &Shape, rhs: &Shape) -> Result<()> {
        if lhs.0 != rhs.0 {
            return Err(CoreError::ShapeMismatch {
                op: "elementwise",
                lhs: lhs.dims().to_vec(),
                rhs: rhs.dims().to_vec(),
            });
        }
        Ok(())
    }
}
