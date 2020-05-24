using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DAL.Entities
{
	public class CnnLayer
	{
		[Key]
		public Guid CnnLayerId { get; set; }

		public int PositionIn { get; set; }

		public int KernelWidth { get; set; }

		public int KernelHeight { get; set; }
		
		public int KernelsCount { get; set; }

		public int FeatureMapsCountIn { get; set; }

		public byte LayerType { get; set; }

		public Guid ModelId { get; set; }

		[ForeignKey("ModelId")]
		public CnnModel Model { get; set; }

		public Guid CnnWeightsId { get; set; }

		[ForeignKey("CnnWeightsId")]
		public CnnWeights Weights { get; set; }
	}
}
