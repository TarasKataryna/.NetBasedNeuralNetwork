using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace FaceRecognitionNN.Helpers
{
    public class FileReaderHelper : IReader
    {
        public List<string[]> Read(string filePath)
        {
            List<string[]> lines = new List<string[]>();
            using (var reader = new StreamReader(filePath))
            {
                while (!reader.EndOfStream)
                {
                    lines.Add(reader.ReadLine().Split(","));
                }
            }

            return lines;
        }

        public void Write(string filePath, List<string> items)
        {
            using (var stream = new StreamWriter(filePath))
            {
                items.ForEach(item => stream.WriteLine(item));
            }
        }
    }
}
