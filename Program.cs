class Program
{
    static void Main()
    {
        string filePath = "Historical Prices/AAPL.csv"; // Your CSV file
        double[] stockPrices = CSVReader.LoadStockPrices(filePath);
        if (stockPrices.Length == 0)
        {
            Console.WriteLine("No data found in the CSV file.");
            return;
        }
        // Normalize Data
        (double[] normalizedPrices, double min, double max) = DataPreprocessing.Normalize(stockPrices);

        // Convert to Sequences
        int sequenceLength = 5; // Use last 5 days to predict next day
        (double[][] sequences, double[] labels) = SequenceGenerator.CreateSequences(normalizedPrices, sequenceLength);

        // Create LSTM Model
        int inputSize = 5;
        int hiddenSize = 10;
        LSTMCell lstm = new LSTMCell(inputSize, hiddenSize);

        // Predict Stock Prices
        foreach (var sequence in sequences)
        {
            double[] output = lstm.Forward(sequence);
            double predictedPrice = DataPreprocessing.Denormalize(output[0], min, max);
            Console.WriteLine($"Predicted Price: {predictedPrice:F2}");
        }
    }
}
