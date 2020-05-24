using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DAL.Entities;

namespace DAL
{
    public class NetworkContext: DbContext
    {
        public NetworkContext() : base("MSConnection") { }

        public DbSet<NetworkModel> NetworkModels { get; set; }

        public DbSet<CnnModel> CnnModels { get; set; }

        public DbSet<CnnLayer> CnnLayers{ get; set; }

        public DbSet<CnnWeights> CnnWeightsSet{ get; set; }

        public DbSet<PerceptronModel> PerceptronModels{ get; set; }

        public DbSet<PerceptronLayer> PerceptronLayers{ get; set; }

        public DbSet<PerceptronWeights> PerceptronWeights{ get; set; }

        protected override void OnModelCreating(DbModelBuilder modelBuilder)
        {

	        modelBuilder.Entity<CnnModel>()
		        .HasOptional(i => i.NetworkModel)
		        .WithOptionalPrincipal(a => a.Cnn);

	        modelBuilder.Entity<PerceptronModel>()
		        .HasOptional(i => i.NetworkModel)
		        .WithOptionalPrincipal(a => a.Perceptron);

	        modelBuilder.Entity<PerceptronWeights>()
		        .HasRequired(i => i.LayerIn)
		        .WithRequiredPrincipal(a => a.Weights);

	        modelBuilder.Entity<CnnWeights>()
		        .HasRequired(i => i.Layer)
		        .WithRequiredDependent(a => a.Weights);


        }

    }
}
