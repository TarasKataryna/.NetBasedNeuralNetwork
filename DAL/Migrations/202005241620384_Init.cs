namespace DAL.Migrations
{
    using System;
    using System.Data.Entity.Migrations;
    
    public partial class Init : DbMigration
    {
        public override void Up()
        {
            CreateTable(
                "dbo.CnnLayers",
                c => new
                    {
                        CnnLayerId = c.Guid(nullable: false),
                        PositionIn = c.Int(nullable: false),
                        KernelWidth = c.Int(nullable: false),
                        KernelHeight = c.Int(nullable: false),
                        KernelsCount = c.Int(nullable: false),
                        FeatureMapsCountIn = c.Int(nullable: false),
                        LayerType = c.Byte(nullable: false),
                        ModelId = c.Guid(nullable: false),
                        CnnWeightsId = c.Guid(nullable: false),
                    })
                .PrimaryKey(t => t.CnnLayerId)
                .ForeignKey("dbo.CnnModels", t => t.ModelId, cascadeDelete: true)
                .Index(t => t.ModelId);
            
            CreateTable(
                "dbo.CnnModels",
                c => new
                    {
                        CnnModelId = c.Guid(nullable: false),
                        Name = c.String(),
                        NetworkModelId = c.Guid(nullable: false),
                    })
                .PrimaryKey(t => t.CnnModelId);
            
            CreateTable(
                "dbo.NetworkModels",
                c => new
                    {
                        NetworkModelId = c.Guid(nullable: false),
                        Name = c.String(),
                        CnnId = c.Guid(nullable: false),
                        PerceptronId = c.Guid(nullable: false),
                        Perceptron_PerceptronModelId = c.Guid(),
                        Cnn_CnnModelId = c.Guid(),
                    })
                .PrimaryKey(t => t.NetworkModelId)
                .ForeignKey("dbo.PerceptronModels", t => t.Perceptron_PerceptronModelId)
                .ForeignKey("dbo.CnnModels", t => t.Cnn_CnnModelId)
                .Index(t => t.Perceptron_PerceptronModelId)
                .Index(t => t.Cnn_CnnModelId);
            
            CreateTable(
                "dbo.PerceptronModels",
                c => new
                    {
                        PerceptronModelId = c.Guid(nullable: false),
                        Name = c.String(),
                        NetworkModelId = c.Guid(nullable: false),
                    })
                .PrimaryKey(t => t.PerceptronModelId);
            
            CreateTable(
                "dbo.PerceptronLayers",
                c => new
                    {
                        PerceptronLayerId = c.Guid(nullable: false),
                        PerceptronModelId = c.Guid(nullable: false),
                        NeuronsCount = c.Int(nullable: false),
                        PositionIn = c.Int(nullable: false),
                    })
                .PrimaryKey(t => t.PerceptronLayerId)
                .ForeignKey("dbo.PerceptronModels", t => t.PerceptronModelId, cascadeDelete: true)
                .ForeignKey("dbo.PerceptronWeights", t => t.PerceptronLayerId)
                .Index(t => t.PerceptronLayerId)
                .Index(t => t.PerceptronModelId);
            
            CreateTable(
                "dbo.PerceptronWeights",
                c => new
                    {
                        PerceptronWeightsId = c.Guid(nullable: false),
                        Height = c.Int(nullable: false),
                        Width = c.Int(nullable: false),
                        Weights = c.String(),
                    })
                .PrimaryKey(t => t.PerceptronWeightsId);
            
            CreateTable(
                "dbo.CnnWeights",
                c => new
                    {
                        CnnWeightsId = c.Guid(nullable: false),
                        LayerId = c.Guid(nullable: false),
                        Weights = c.String(),
                    })
                .PrimaryKey(t => t.CnnWeightsId)
                .ForeignKey("dbo.CnnLayers", t => t.CnnWeightsId)
                .Index(t => t.CnnWeightsId);
            
        }
        
        public override void Down()
        {
            DropForeignKey("dbo.CnnWeights", "CnnWeightsId", "dbo.CnnLayers");
            DropForeignKey("dbo.NetworkModels", "Cnn_CnnModelId", "dbo.CnnModels");
            DropForeignKey("dbo.NetworkModels", "Perceptron_PerceptronModelId", "dbo.PerceptronModels");
            DropForeignKey("dbo.PerceptronLayers", "PerceptronLayerId", "dbo.PerceptronWeights");
            DropForeignKey("dbo.PerceptronLayers", "PerceptronModelId", "dbo.PerceptronModels");
            DropForeignKey("dbo.CnnLayers", "ModelId", "dbo.CnnModels");
            DropIndex("dbo.CnnWeights", new[] { "CnnWeightsId" });
            DropIndex("dbo.PerceptronLayers", new[] { "PerceptronModelId" });
            DropIndex("dbo.PerceptronLayers", new[] { "PerceptronLayerId" });
            DropIndex("dbo.NetworkModels", new[] { "Cnn_CnnModelId" });
            DropIndex("dbo.NetworkModels", new[] { "Perceptron_PerceptronModelId" });
            DropIndex("dbo.CnnLayers", new[] { "ModelId" });
            DropTable("dbo.CnnWeights");
            DropTable("dbo.PerceptronWeights");
            DropTable("dbo.PerceptronLayers");
            DropTable("dbo.PerceptronModels");
            DropTable("dbo.NetworkModels");
            DropTable("dbo.CnnModels");
            DropTable("dbo.CnnLayers");
        }
    }
}
