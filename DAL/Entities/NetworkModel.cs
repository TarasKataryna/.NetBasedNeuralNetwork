using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace DAL.Entities
{
	public class NetworkModel
	{
		[Key]
		public Guid NetworkModelId { get; set; }

		public string Name { get; set; }

		public Guid CnnId { get; set; }

		public CnnModel Cnn { get; set; }

		public Guid PerceptronId { get; set; }
		
		public PerceptronModel Perceptron { get; set; }
	}
}
