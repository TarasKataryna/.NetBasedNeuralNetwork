using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DAL.Entities
{
	public class CnnWeights
	{
		[Key]
		public Guid CnnWeightsId { get; set; }

		public Guid LayerId { get; set; }

		public CnnLayer Layer { get; set; }

		public string Weights { get; set; }
	}
}
