using System;

class Program
{
    static void Main()
    {
        // Step 1: Load and preprocess data
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
        int inputSize = sequenceLength;
        int hiddenSize = 10;
        LSTMCell lstm = new LSTMCell(inputSize, hiddenSize);

        // Training parameters
        int epochs = 100;
        double learningRate = 0.01;

        // Step 2: Train the model
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalLoss = 0;

            for (int i = 0; i < sequences.Length; i++)
            {
                // Forward pass
                double[] output = lstm.Forward(sequences[i]);

                // Compute loss (e.g., Mean Squared Error)
                double loss = ComputeMeanSquaredError(output[0], labels[i]);
                totalLoss += loss;

                // Compute gradients
                double[] dLoss_dh = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++)
                {
                    dLoss_dh[j] = 2 * (output[j] - labels[i]); // Derivative of MSE
                }
                double[] dLoss_dc_next = new double[hiddenSize]; // Assume zero for the last time step
                var (dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo) = lstm.Backward(dLoss_dh, dLoss_dc_next);

                // Update weights and biases using gradient descent
                lstm.UpdateWeights(dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo, learningRate);
            }

            Console.WriteLine($"Epoch {epoch + 1}, Loss: {totalLoss / sequences.Length}");
        }

        // Step 3: Test the model
        Console.WriteLine("\nTesting the model...");
        for (int i = 0; i < sequences.Length; i++)
        {
            double[] output = lstm.Forward(sequences[i]);
            double predictedPrice = DataPreprocessing.Denormalize(output[0], min, max);
            double actualPrice = DataPreprocessing.Denormalize(labels[i], min, max);
            Console.WriteLine($"Actual Price: {actualPrice:F2}, Predicted Price: {predictedPrice:F2}");
        }
    }

    static double ComputeMeanSquaredError(double predicted, double actual)
    {
        return Math.Pow(predicted - actual, 2);
    }
}