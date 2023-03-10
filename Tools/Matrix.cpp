//
// Created by Vladimir Smirnov on 11.09.2021.
//

#include "Matrix.h"
#include <fstream>
#include <cassert>

Matrix::Matrix(size_t rows, size_t columns) : _rows(rows), _columns(columns) {
    data.resize(rows*columns);
}

Matrix::Matrix(size_t rows, size_t columns, scalar value) : _rows(rows), _columns(columns) {
    data.resize(rows*columns, value);
}

Matrix::Matrix(size_t rows, size_t columns, string path) : _rows(rows), _columns(columns) {
    data.resize(rows*columns);
    ifstream input(path);
    if (input) {
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                input >> data[row * _columns + column];
            }
        }
    }
}

int Matrix::rows() const { return _rows; }

int Matrix::columns() const { return _columns; }

scalar * Matrix::data_ptr() { return data.data(); }

const scalar * Matrix::data_const_ptr() const { return data.data(); }

scalar& Matrix::operator()(size_t row, size_t column) {
    if (row >= _rows or column >= _columns) {
        cout << "Error, index (" << row << " " << column << ") out of range (return NULL)" << endl;
        throw;
    }
    return data[row * _columns + column];
}

scalar Matrix::operator()(size_t row, size_t column) const {
    if (row >= _rows or column >= _columns) {
        cout << "Error, index (" << row << " " << column << ") out of range (return 1st element)" << endl;
        throw;
    }
    return data[row * _columns + column];
}

void Matrix::print() const {
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _columns; j++) {
            cout << this->operator()(i, j) << " ";
        }
        cout << endl;
    }
}

void Matrix::print_to_file(const string& pth){
    ofstream fout(pth);
    for (int row = 0; row < _rows; ++row){
        for (int col = 0; col < _columns; ++col){
            fout << this->operator()(row, col) << ' ';
        }
        fout << endl; 
    }
} 

void Matrix::fill(const scalar value) {
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _columns; j++) {
            this->operator()(i, j) = value;
        }
    }
}

void Matrix::copy(Matrix& matrix) {
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _columns; j++) {
            this->operator()(i, j) = matrix(i, j);
            //data[i*_columns+j] = matrix.data_ptr()[i*_columns+j];
        }
    }
}

void Matrix::resize(size_t rows, size_t columns) {
    data.resize(rows*columns);
    _rows = rows;
    _columns = columns;
}

Matrix Matrix::operator=(const Matrix& newOne)
{
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _columns; j++) {
            this->operator()(i, j) = newOne(i, j);
        }
    }
    return *this;
}

Matrix Matrix::operator+(const Matrix &other) const {
    Matrix tmp(_rows, _columns);
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _columns; j++) {
            tmp(i, j) = this->operator()(i, j) + other(i, j);
        }
    }
    return tmp;
}

Matrix Matrix::operator-(const Matrix &other) const {
    Matrix tmp(_rows, _columns);
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _columns; j++) {
            tmp(i, j) = this->operator()(i, j) - other(i, j);
        }
    }
    return tmp;
}

std::ostream& operator<<(std::ostream &out, const Matrix &matrix) {
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.columns(); j++) {
            out << matrix(i, j) << " ";
        }
        out << endl;
    }
    return out;
}

Matrix Matrix::operator/(scalar value) const {
    Matrix tmp(_rows, _columns);
    assert(value != 0);
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _columns; j++) {
            tmp(i, j) = this->operator()(i, j) / value;
        }
    }
    return tmp;
}

Matrix Matrix::operator*(scalar value) const {
    Matrix tmp(_rows, _columns);
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _columns; j++) {
            tmp(i, j) = this->operator()(i, j) * value;
        }
    }
    return tmp;
}

Matrix& Matrix::operator+=(const Matrix &other) {
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _columns; j++) {
            this->operator()(i, j) += other(i, j);
        }
    }
    return *this;
}

Matrix &Matrix::operator-=(const Matrix &other) {
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _columns; j++) {
            this->operator()(i, j) -= other(i, j);
        }
    }
    return *this;
}

Matrix &Matrix::operator/=(scalar value) {
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _columns; j++) {
            this->operator()(i, j) /= value;
        }
    }
    return *this;
}

vector<scalar> Matrix::ToVector()
{
    return data;
}

void Matrix::InitFromArray(scalar matrix[])
{
    for (int i = 0; i < _rows * _columns; ++i)
    {
        data[i] = matrix[i];
    }
}

void Matrix::InitFromVector(const vector<scalar>& matrix)
{
    data.clear();
    data = matrix;
}

vector<scalar> Matrix::StripToVector(int cellX)
{
    vector<scalar> stripVec;
    stripVec.resize(_rows);

    for (int i = 0; i < _rows; ++i)
    {
        stripVec[i] = data[i * _columns + cellX];
    }

    return stripVec;
}

void Matrix::SetData(int row, int col, scalar value)
{
    data[row * _columns + col] = value;
}