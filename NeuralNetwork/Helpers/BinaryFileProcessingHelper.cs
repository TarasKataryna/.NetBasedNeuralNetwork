using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Helpers
{
    public static class BinaryFileProcessingHelper
    {
        public static async Task<Tuple<int, List<double[][]>>> GetInputData(string filePath, int idx)
        {
            var inputData =  new List<double[][]>();
            var inputResult = 0;

            using (var fileStream = File.OpenRead(filePath))
            {
                int required = 3073; //1 bytes for class, 1024 bytes for red, 1024 bytes for green, 1024 bytes for blue
                int position = idx * required;

                fileStream.Seek(position, SeekOrigin.Begin);

                byte[] buffer = new byte[required];

                await fileStream.ReadAsync(buffer, 0, required);

                inputResult = Convert.ToInt32(buffer[0]);

                var redMatrix = new double[32][];
                var greenMatrix = new double[32][];
                var blueMatrix = new double[32][];

                Bitmap bitmap = new Bitmap(32, 32);

                for (int i = 0; i < 32; ++i)
                {
                    redMatrix[i] = new double[32];
                    greenMatrix[i] = new double[32];
                    blueMatrix[i] = new double[32];

                    for(int j = 0; j < 32; ++j)
                    {
                        redMatrix[i][j] = Convert.ToDouble(buffer[1 + i * 32 + j]) / 255 * 0.99 + 0.01;
                        greenMatrix[i][j] = Convert.ToDouble(buffer[1025 + i * 32 + j]) / 255 * 0.99 + 0.01;
                        blueMatrix[i][j] = Convert.ToDouble(buffer[2049 + i * 32 + j]) / 255 * 0.99 + 0.01;
                    }

                    //for (int j = 0; j < 32; ++j)
                    //{
                    //    var r  = buffer[1 + i * 32 + j];
                    //    var g = buffer[1025 + i * 32 + j];
                    //    var b = buffer[2049 + i * 32 + j];
                    //    bitmap.SetPixel(i, j, Color.FromArgb(255,r,g,b));
                    //}
                }

                inputData.Add(redMatrix);
                inputData.Add(greenMatrix);
                inputData.Add(blueMatrix);

                //bitmap.Save("D:\\test.jpg", ImageFormat.Jpeg);
            }

            return new Tuple<int, List<double[][]>>(inputResult, inputData);
        }
    }
}
