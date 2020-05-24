using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DAL.Entities
{
	public class PerceptronLayer
	{
		[Key]
		public Guid PerceptronLayerId { get; set; }

		public Guid PerceptronModelId { get; set; }

		[ForeignKey("PerceptronModelId")]
		public PerceptronModel Perceptron { get; set; }

		public int NeuronsCount { get; set; }

		public int PositionIn { get; set; }

		public PerceptronWeights Weights { get; set; }
	}
}
