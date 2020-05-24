using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DAL.Entities
{
	public class PerceptronModel
	{
		[Key]
		public Guid PerceptronModelId { get; set; }

		public string Name { get; set; }

		public Guid NetworkModelId { get; set; }

		public NetworkModel NetworkModel { get; set; }

		public virtual List<PerceptronLayer> Layers { get; set; }
	}
}
