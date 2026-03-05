mod args;
mod qsar;
use crate::args::CommandParse;
use crate::args::Commands;
use crate::qsar::qsar_chem;
use clap::Parser;
use figlet_rs::FIGfont;
use smartcore::linear::logistic_regression::LogisticRegression;
mod smile;
use crate::smile::readsmiles;
use crate::smile::readsmiles_predict;
use smartcore::metrics::accuracy;
use smartcore::model_selection::train_test_split;
use std::fs::File;
use std::io::Write;
mod knn;
use crate::knn::classifyadd;
mod xgb;
use crate::xgb::classifyadd_knn;

/*
Gaurav Sablok
codeprog@icloud.com
*/

fn main() {
    let fontgenerate = FIGfont::standard().unwrap();
    let repgenerate = fontgenerate.convert("beechem");
    println!("{}", repgenerate.unwrap());

    let args = CommandParse::parse();
    match &args.command {
        Commands::QSARElasticNet {
            filepath,
            nfold,
            thread,
        } => {
            let n_threads = thread.parse::<usize>().expect("thread must be a number");
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("failed to create thread pool");
            pool.install(|| {
                let moduleqqsar = qsar_chem(filepath, nfold).unwrap();

                println!("The qsar modelling has finished: {}", moduleqqsar);
            });
        }
        Commands::LogisticSMILE {
            smiles,
            expression,
            sequencefile,
            threads,
            threshold,
            predexp,
            predseq,
            predsmiles,
        } => {
            let n_threads = threads.parse::<usize>().unwrap();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("file not present");
            pool.install(|| {
                let valueinput = readsmiles(smiles, sequencefile, expression, threshold).unwrap();
                let logisticvalue =
                    LogisticRegression::fit(&valueinput.0, &valueinput.1, Default::default())
                        .unwrap();
                let predvalue = readsmiles_predict(predsmiles, predseq, predexp).unwrap();
                let valuepred = logisticvalue.predict(&predvalue).unwrap();
                let accuracy_value = accuracy(&valueinput.1, &valuepred);
                println!(
                    "The accuracy value on the entire training datasets is {}",
                    accuracy_value
                );

                let splitratio =
                    train_test_split(&valueinput.0, &valueinput.1, 0.2, true, Some(2811));
                let logistictest =
                    LogisticRegression::fit(&splitratio.0, &splitratio.2, Default::default())
                        .unwrap();
                let logisticpredict = logistictest.predict(&splitratio.1).unwrap();
                let accuracytestsplit = accuracy(&splitratio.3, &logisticpredict);
                println!(
                    "The  accuracy of the predicted values on the split basis is: {}",
                    accuracytestsplit
                );
                let splitpredict = logistictest.predict(&predvalue).unwrap();
                let mut filewrite = File::create("splitpredict.txt").expect("file not present");
                for i in splitpredict.iter() {
                    writeln!(filewrite, "{}", i).expect("line not present");
                }

                let value = classifyadd(
                    smiles,
                    expression,
                    sequencefile,
                    threshold,
                    predsmiles,
                    predseq,
                    predexp,
                )
                .unwrap();
                println!("The random classifier has finished:{}", value);
            });

            let knnclassify = classifyadd_knn(
                smiles,
                expression,
                sequencefile,
                threshold,
                predsmiles,
                predseq,
                predexp,
            )
            .unwrap();
            println!("The KNN classifer has finished:{}", knnclassify);
        }
    }
}
