using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DAL.Entities
{
	public class PerceptronWeights
	{
		[Key, ForeignKey("LayerIn")]
		public Guid PerceptronWeightsId { get; set; }

		public PerceptronLayer LayerIn { get; set; }

		public int Height { get; set; }

		public int Width { get; set; }

		public string Weights { get; set; }

	}
}
