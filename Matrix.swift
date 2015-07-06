// Matrix.swift
// A lot of things borrowed from https://github.com/mattt/Surge/blob/master/Source/
// see Mattt Thompson (http://mattt.me)

import Accelerate

public struct Matrix<T where T: FloatingPointType, T: FloatLiteralConvertible> {
    typealias Element = T
    
    let rows: Int
    let columns: Int
    var grid: [Element]
    
    public init(rows: Int, columns: Int, repeatedValue: Element) {
        self.rows = rows
        self.columns = columns
        
        self.grid = [Element](count: rows * columns, repeatedValue: repeatedValue)
    }
    
    public init(_ contents: [[Element]]) {
        let m: Int = contents.count
        let n: Int = contents[0].count
        let repeatedValue: Element = 0.0
        
        self.init(rows: m, columns: n, repeatedValue: repeatedValue)
        
        for (i, row) in enumerate(contents) {
            grid.replaceRange(i*n..<i*n+min(m, row.count), with: row)
        }
    }
    
    // Initialize directly with a grid
    public init(grid: [Element], rows: Int, columns: Int) {
        self.grid = grid
        self.rows = rows
        self.columns = columns
    }
    
    // Initialize a zero matrix with the provided vector as diagonale
    public init(diagonalMatrixwithVector: [Element]) {
        
        self.init(rows: diagonalMatrixwithVector.count, columns: diagonalMatrixwithVector.count, repeatedValue: 0.0)
        
        for i in 0...diagonalMatrixwithVector.count-1 {
            self[i,i] = diagonalMatrixwithVector[i];
        }
    
    }
    
    // Access functions
    public subscript(row: Int, column: Int) -> Element {
        get {
            assert(indexIsValidForRow(row, column: column))
            return grid[(row * columns) + column]
        }
        
        set {
            assert(indexIsValidForRow(row, column: column))
            grid[(row * columns) + column] = newValue
        }
    }
    
    private func indexIsValidForRow(row: Int, column: Int) -> Bool {
        return row >= 0 && row < rows && column >= 0 && column < columns
    }
}

// MARK: - Printable

extension Matrix: Printable {
    public var description: String {
        var description = ""
        
        for i in 0..<rows {
            let contents = join("\t", map(0..<columns){"\(self[i, $0])"})
            
            switch (i, rows) {
            case (0, 1):
                description += "\(contents)\t"
            case (0, _):
                description += "\(contents)\t"
            case (rows - 1, _):
                description += "\(contents)\t"
            default:
                description += "\(contents)\t"
            }
            
            description += "\n"
        }
        
        return description
    }
}

// MARK: - SequenceType

extension Matrix: SequenceType {
    public func generate() -> GeneratorOf<ArraySlice<Element>> {
        let endIndex = rows * columns
        var nextRowStartIndex = 0
        
        return GeneratorOf<ArraySlice<Element>> {
            if nextRowStartIndex == endIndex {
                return nil
            }
            
            let currentRowStartIndex = nextRowStartIndex
            nextRowStartIndex += self.columns
            
            return self.grid[currentRowStartIndex..<nextRowStartIndex]
        }
    }
}

// MARK: -

public func add(x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with addition")
    
    var results = y
    cblas_daxpy(Int32(x.grid.count), 1.0, x.grid, 1, &(results.grid), 1)
    
    return results
}

public func mul(alpha: Double, x: Matrix<Double>) -> Matrix<Double> {
    var results = x
    cblas_dscal(Int32(x.grid.count), alpha, &(results.grid), 1)
    
    return results
}

public func mul(x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.columns == y.rows, "Matrix dimensions not compatible with multiplication")
    
    var results = Matrix<Double>(rows: x.rows, columns: y.columns, repeatedValue: 0.0)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(x.rows), Int32(y.columns), Int32(x.columns), 1.0, x.grid, Int32(x.columns), y.grid, Int32(y.columns), 0.0, &(results.grid), Int32(results.columns))
    
    return results
}

public func inv(x : Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == x.columns, "Matrix must be square")
    
    var results = x
    
    var ipiv = [__CLPK_integer](count: x.rows * x.rows, repeatedValue: 0)
    var lwork = __CLPK_integer(x.columns * x.columns)
    var work = [CDouble](count: Int(lwork), repeatedValue: 0.0)
    var error: __CLPK_integer = 0
    var nc = __CLPK_integer(x.columns)
    
    dgetrf_(&nc, &nc, &(results.grid), &nc, &ipiv, &error)
    dgetri_(&nc, &(results.grid), &nc, &ipiv, &work, &lwork, &error)
    
    assert(error == 0, "Matrix not invertible")
    
    return results
}

public func transpose(x: Matrix<Float>) -> Matrix<Float> {
    var results = Matrix<Float>(rows: x.columns, columns: x.rows, repeatedValue: 0.0)
    vDSP_mtrans(x.grid, 1, &(results.grid), 1, vDSP_Length(results.rows), vDSP_Length(results.columns))
    
    return results
}

public func transpose(x: Matrix<Double>) -> Matrix<Double> {
    var results = Matrix<Double>(rows: x.columns, columns: x.rows, repeatedValue: 0.0)
    vDSP_mtransD(x.grid, 1, &(results.grid), 1, vDSP_Length(results.rows), vDSP_Length(results.columns))
    return results
}



// MARK: - Operators

public func + (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return add(lhs, rhs)
}

public func * (lhs: Double, rhs: Matrix<Double>) -> Matrix<Double> {
    return mul(lhs, rhs)
}

public func * (lhs: Double, rhs: [Double]) -> [Double] {
    var res = rhs;
    for i in 0...rhs.count-1 {
        res[i] = res[i] * lhs;
    }
    return res;
}

public func * (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return mul(lhs, rhs)
}

postfix operator ′ {}

public postfix func ′ (value: Matrix<Double>) -> Matrix<Double> {
    return transpose(value)
}

public func mean(x: [Double]) -> Double {
    var result: Double = 0.0
    vDSP_meanvD(x, 1, &result, vDSP_Length(x.count))
    return result
}

public func add(x: [Double], y: [Double]) -> [Double] {
    var results = [Double](y)
    cblas_daxpy(Int32(x.count), 1.0, x, 1, &results, 1)
    return results
}

func + (lhs: [Double], rhs: Double) -> [Double] {
    return add(lhs, [Double](count: lhs.count, repeatedValue: rhs))
}

/**
* Centers mat M. (Subtracts the mean from every column)
*/

public func centerColumns(x: Matrix<Double>) -> (x: Matrix<Double>, meanValues: [Double]) {
    
    // Get rows & cols
    var rows = x.rows;
    var columns = x.columns;
    
    // Create output variables
    var meanValues = [Double](count: columns, repeatedValue: 0.0)
    var outMatrix = [Double](count: columns*rows, repeatedValue: 0.0)
    
    // Transpose first so we can access the columns easier
    var xT = transpose(x);
    
    // Select values based on range, compute mean, and replace the original values
    for i in 0...columns-1 {
        let range = rows*i..<rows*(i+1)
        var currentColumn : [Double] = Array(xT.grid[range])
        var result = -mean(currentColumn)
        currentColumn = currentColumn+result
        meanValues[i] = -result
        xT.grid.replaceRange(range, with: currentColumn)
    }
    
    // Transpose back
    return (transpose(xT), meanValues)
}

/**
* DeCenters mat M. (Adds the mean to every column according to the vector provided)
*/

public func deCenterColumns(x: Matrix<Double>, meanValues: [Double]) -> Matrix<Double> {
    
    assert(x.columns == meanValues.count)
    
    var result = x
    
    for i in 0...x.rows-1 {
        for j in 0...x.columns-1 {
            result[i,j] = meanValues[j]
        }
    }
    
    return x
}


/**
* Computes the mean of every row.
*/

public func meanOfRows(x: Matrix<Double>) -> [Double]
{
    // Get rows & cols
    var rows = x.rows;
    var columns = x.columns;
    var sum = 0.0
    
    var meanValues = [Double](count: rows, repeatedValue: 0.0)
    
    for i in 0...rows-1 {
        sum = 0.0
        for j in 0...columns-1 {
            sum += x[i,j]
        }
        meanValues[i] = sum/Double(columns)
    }
    
    return meanValues
}

/**
* Returns the maximal element on the diagonal
* of the matrix M.
*/

public func maxDiag(x: Matrix<Double>) -> Double
{
    var max = x[0,0];
    
    for i in 1...x.rows-1 {
        if (x[i,i] > max) {
            max = x[i,i];
        }
    }
    
    return max;
}

/**
* Applyies function fx() with parameter par on
* every matrix element.
*/

public func applyFunc(x: Matrix<Double>, par: Double, function: (Double, Double) -> Double) -> Matrix<Double> {
    
    var result = x
    
    for i in 0...(result.rows*result.columns)-1 {
        result.grid[i] = function(result.grid[i], par)
    }
    
    return result
    
    
}

/**
* Applyies function fx() with parameter par on
* every vector element.
*/

public func applyFunc(x: [Double], par: Double, function: (Double, Double) -> Double) -> [Double] {
    
    var result = x
    
    for i in 0...x.count-1 {
        result[i] = function(result[i], par)
    }
    
    return result
    
    
}