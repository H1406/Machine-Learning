public class DataPreprocessing
{
    public static (double[],double,double) Normalize(double[] data)
    {
        double min = double.MaxValue;
        double max = double.MinValue;

        foreach (double value in data)
        {
            if (value < min) min = value;
            if (value > max) max = value;
        }
        double[] normalizedData = new double[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            normalizedData[i] = (data[i] - min) / (max - min);
        }
        return (normalizedData, min, max);
    }
    public static double Denormalize(double data, double min, double max){
        return data*(max-min)+min;
    }
}
class SequenceGenerator
{
    public static (double[][], double[]) CreateSequences(double[] data, int sequenceLength)
    {
        List<double[]> sequences = new List<double[]>();
        List<double> labels = new List<double>();

        for (int i = 0; i < data.Length - sequenceLength; i++)
        {
            double[] sequence = new double[sequenceLength];
            Array.Copy(data, i, sequence, 0, sequenceLength);
            sequences.Add(sequence);
            labels.Add(data[i + sequenceLength]); // Target value
        }

        return (sequences.ToArray(), labels.ToArray());
    }
}
