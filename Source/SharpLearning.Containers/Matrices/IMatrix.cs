
namespace SharpLearning.Containers.Matrices
{
    /// <summary>
    /// Matrix interface
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IMatrix<T>
    {
        /// <summary>
        /// Gets item at location (row, col)
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <returns></returns>
        T GetItemAt(int row, int col);

        /// <summary>
        /// Sets item at location(row, col)
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <param name="item"></param>
        void SetItemAt(int row, int col, T item);

        /// <summary>
        /// Gets row at index
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        T[] GetRow(int index);

        /// <summary>
        /// Gets column at index
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        T[] GetColumn(int index);

        /// <summary>
        /// Gets row at index 
        /// </summary>
        /// <param name="index"></param>
        /// <param name="row"></param>
        void GetRow(int index, T[] row);

        /// <summary>
        /// Gets column at index
        /// </summary>
        /// <param name="index"></param>
        /// <param name="col"></param>
        void GetColumn(int index, T[] col);

        /// <summary>
        /// Gets rows
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        IMatrix<T> GetRows(params int[] indices);

        /// <summary>
        /// Gets columns
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        IMatrix<T> GetColumns(params int[] indices);

        /// <summary>
        /// Gets the array which stores all matrix values. values are storred row-wise.
        /// </summary>
        /// <returns></returns>
        T[] Data();

        /// <summary>
        /// Gets the number of columns
        /// </summary>
        /// <returns></returns>
        int GetNumberOfColumns();

        /// <summary>
        /// Gets the numver of rows
        /// </summary>
        /// <returns></returns>
        int GetNumberOfRows();
    }
}
