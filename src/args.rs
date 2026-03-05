use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "Beechem",
    version = "1.0",
    about = "QSAR modelling of the chemical compounds
       ************************************************
       Gaurav Sablok,
       Email: codeprog@icloud.com
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// estimation sheet
    QSARElasticNet {
        /// path to the file
        filepath: String,
        /// nfold value
        nfold: String,
        /// threads for the analysis
        thread: String,
    },
    /// logistic classifier for smiles
    LogisticSMILE {
        /// path to the smiles chmeical uses
        smiles: String,
        /// path to the expression file
        expression: String,
        /// sequence file
        sequencefile: String,
        /// threads for the analysis
        threads: String,
        /// expressio threashold for the classification
        threshold: String,
        /// prediction sequence file
        predseq: String,
        /// prediction expression file
        predexp: String,
        /// pred smiles file
        predsmiles: String,
    },
}
