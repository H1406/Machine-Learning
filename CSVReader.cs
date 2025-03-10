using System;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
public class CSVReader{
    public static double[] LoadStockPrices(string filePath){
        List<double>prices = new List<double>();
        using(var reader = new StreamReader(filePath)){
            string headerLine = reader.ReadLine();
            while (!reader.EndOfStream){
                string line = reader.ReadLine();
                string[] values = line.Split(',');
                string PriceStr = values[4].Trim('"');
                double price = double.Parse(PriceStr, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture);
                prices.Add(price);
            }
        }
        return prices.ToArray();
    }
}