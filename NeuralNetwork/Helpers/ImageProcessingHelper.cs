﻿using System;
using System.Collections.Generic;
using System.Text;

using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Drawing2D;


namespace NeuralNetwork.Helpers
{
    public static class ImageProcessingHelper
    {
        public static List<double[][]> PrepareData(string filePath)
        {
            var image = (Bitmap)Image.FromFile(filePath);

            int height = image.Height;
            int width = image.Width;

            var R = ArrayHelper.ZeroMatrix(height, width);
            var G = ArrayHelper.ZeroMatrix(height, width);
            var B = ArrayHelper.ZeroMatrix(height, width);

            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    R[i][j] = image.GetPixel(j, i).R;
                    G[i][j] = image.GetPixel(j, i).G;
                    B[i][j] = image.GetPixel(j, i).B;
                }
            }

            var result = new List<double[][]>();
            result.Add(R);
            result.Add(G);
            result.Add(B);

            return result;
        }
    }
}