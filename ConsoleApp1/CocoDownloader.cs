using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Threading.Tasks;
using System.IO;
using System.Net;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Drawing2D;

namespace Downloader
{
    public class CocoDownloader
    {
        public async Task Download(string jsonPath, string downloadInfo, string resultInfo, CancellationToken token)
        {
            var str = File.ReadAllText(jsonPath);
            var images = JArray.Parse(str);

            var lastId = new FileInfo(resultInfo).Length == 0 
                ? images.First.Value<string>("id") 
                : File.ReadAllText(resultInfo);

            var imagesList = images.AsJEnumerable().ToList();
            var index = imagesList.ToList().FindIndex(item => item.Value<string>("id") == lastId);

            try
            {
                using (HttpClient client = new HttpClient())
                {
                    lastId = "";
                    var strLastId = string.Empty;
                    var count = imagesList.Count;
                    for (int i = index + 1; i < count; ++i)
                    {
                        try
                        {
                            var fileName = $"{downloadInfo}\\{imagesList[i].Value<string>("file_name")}";
                            var response = await client.GetAsync(new Uri(imagesList[i].Value<string>("coco_url")));

                            if (response.IsSuccessStatusCode)
                            {
                                var res = await response.Content.ReadAsByteArrayAsync();
                                File.WriteAllBytes(fileName, res);
                                lastId = imagesList[i].Value<string>("id");
                            }

                            if (token.IsCancellationRequested)
                            {
                                break;
                            }
                        }
                        catch(Exception ex)
                        {

                        }
                    }
                }
            }
            finally
            {
                File.WriteAllText(resultInfo, lastId.ToString());
            }

        }

        public async Task Resize(string pathFrom, string pathTo, int size, CancellationToken token)
        {
            string[] fileNames = Directory.GetFiles(pathFrom);

            for(int i = 0; i < fileNames.Length; ++i)
            {
                Stream imageStream = new MemoryStream(File.ReadAllBytes(fileNames[i]));
                Image image = Image.FromStream(imageStream);

                var destRect = new Rectangle(0, 0, size, size);
                var destImage = new Bitmap(size, size);

                destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

                using (var graphics = Graphics.FromImage(destImage))
                {
                    graphics.CompositingMode = CompositingMode.SourceCopy;
                    graphics.CompositingQuality = CompositingQuality.HighQuality;
                    graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    graphics.SmoothingMode = SmoothingMode.HighQuality;
                    graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                    using (var wrapMode = new ImageAttributes())
                    {
                        wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                        graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                    }

                    destImage.Save(fileNames[i]);
                }

                if (token.IsCancellationRequested)
                {
                    return;
                }
            }
        }
    }
}
