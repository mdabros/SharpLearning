
namespace SharpLearning.Containers.Matrices
{
    public interface IMatrix<T>
    {
        T GetItemAt(int row, int col);
        void SetItemAt(int row, int col, T item);

        T[] GetRow(int index);
        T[] GetColumn(int index);

        IMatrix<T> GetRows(int[] indices);
        IMatrix<T> GetColumns(int[] indices);

        T[] GetFeatureArray();
        int GetNumberOfColumns();
        int GetNumberOfRows();
    }
}
