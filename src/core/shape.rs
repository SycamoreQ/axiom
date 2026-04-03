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

    pub fn scalar() -> Self {
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
        self.0.iter().product()
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

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Self {
        Self::new(&v)
    }
}

impl From<(usize, usize)> for Shape {
    fn from((a, b): (usize, usize)) -> Self {
        Self::new(&[a, b])
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from((a, b, c): (usize, usize, usize)) -> Self {
        Self::new(&[a, b, c])
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from((a, b, c, d): (usize, usize, usize, usize)) -> Self {
        Self::new(&[a, b, c, d])
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_dims() {
        let s = Shape::new(&[2, 3, 4]);
        assert_eq!(s.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_rank() {
        assert_eq!(Shape::new(&[2, 3]).rank(), 2);
        assert_eq!(Shape::new(&[2, 3, 4, 5]).rank(), 4);
        assert_eq!(Shape::scalar().rank(), 0);
    }

    #[test]
    fn test_numel() {
        assert_eq!(Shape::new(&[2, 3, 4]).numel(), 24);
        assert_eq!(Shape::new(&[10, 10]).numel(), 100);
        assert_eq!(Shape::scalar().numel(), 1);
    }

    #[test]
    fn test_is_scalar_vector_matrix() {
        assert!(Shape::scalar().is_scalar());
        assert!(Shape::new(&[5]).is_vector());
        assert!(Shape::new(&[3, 3]).is_matrix());
        assert!(!Shape::new(&[3, 3]).is_vector());
    }

    #[test]
    fn test_dim_valid() {
        let s = Shape::new(&[4, 8, 16]);
        assert_eq!(s.dim(0).unwrap(), 4);
        assert_eq!(s.dim(2).unwrap(), 16);
    }

    #[test]
    fn test_dim_out_of_bounds() {
        let s = Shape::new(&[4, 8]);
        assert!(s.dim(5).is_err());
    }

    #[test]
    fn test_matmul_check_valid() {
        let lhs = Shape::new(&[4, 8]);
        let rhs = Shape::new(&[8, 16]);
        let result = Shape::matmul_check(&lhs, &rhs).unwrap();
        assert_eq!(result.dims(), &[4, 16]);
    }

    #[test]
    fn test_matmul_check_inner_mismatch() {
        let lhs = Shape::new(&[4, 8]);
        let rhs = Shape::new(&[9, 16]);
        assert!(Shape::matmul_check(&lhs, &rhs).is_err());
    }

    #[test]
    fn test_matmul_check_rank_too_low() {
        let lhs = Shape::new(&[4]);
        let rhs = Shape::new(&[4, 8]);
        assert!(Shape::matmul_check(&lhs, &rhs).is_err());
    }

    #[test]
    fn test_matmul_check_batched() {
        let lhs = Shape::new(&[2, 4, 8]);
        let rhs = Shape::new(&[2, 8, 16]);
        let result = Shape::matmul_check(&lhs, &rhs).unwrap();
        assert_eq!(result.dims(), &[4, 16]);
    }

    #[test]
    fn test_elementwise_check_valid() {
        let a = Shape::new(&[2, 3, 4]);
        let b = Shape::new(&[2, 3, 4]);
        assert!(Shape::elementwise_check(&a, &b).is_ok());
    }

    #[test]
    fn test_elementwise_check_mismatch() {
        let a = Shape::new(&[2, 3]);
        let b = Shape::new(&[2, 4]);
        assert!(Shape::elementwise_check(&a, &b).is_err());
    }

    #[test]
    fn test_from_tuple_2d() {
        let s = Shape::from((4usize, 8usize));
        assert_eq!(s.dims(), &[4, 8]);
    }

    #[test]
    fn test_from_tuple_3d() {
        let s = Shape::from((2usize, 4usize, 8usize));
        assert_eq!(s.dims(), &[2, 4, 8]);
    }

    #[test]
    fn test_from_vec() {
        let s = Shape::from(vec![1usize, 2, 3]);
        assert_eq!(s.dims(), &[1, 2, 3]);
    }

    #[test]
    fn test_display() {
        let s = Shape::new(&[2, 3, 4]);
        assert_eq!(format!("{}", s), "[2, 3, 4]");
    }

    #[test]
    fn test_rows_cols() {
        let s = Shape::new(&[4, 8]);
        assert_eq!(s.rows().unwrap(), 4);
        assert_eq!(s.cols().unwrap(), 8);
    }

    #[test]
    fn test_rows_cols_vector_fails() {
        let s = Shape::new(&[4]);
        assert!(s.rows().is_err());
        assert!(s.cols().is_err());
    }
}
