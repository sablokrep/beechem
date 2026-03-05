use crate::smile::readsmiles;
use crate::smile::readsmiles_predict;
use smartcore::metrics::accuracy;
use smartcore::model_selection::train_test_split;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use std::error::Error;
use std::fs::File;
use std::io::Write;

/*
Gaurav Sablok
codeprog@icloud.com
*/

pub fn classifyadd_knn(
    path1: &str,
    path2: &str,
    path3: &str,
    path4: &str,
    predfile: &str,
    predseq: &str,
    predexp: &str,
) -> Result<String, Box<dyn Error>> {
    let valueinput = readsmiles(path1, path2, path3, path4).unwrap();
    let randomvalue = KNNClassifier::fit(&valueinput.0, &valueinput.1, Default::default()).unwrap();
    let predvalue = readsmiles_predict(predfile, predseq, predexp).unwrap();
    let valuepred = randomvalue.predict(&predvalue).unwrap();
    let accuracy_value = accuracy(&valueinput.1, &valuepred);
    println!(
        "The accuracy value on the entire training datasets is {}",
        accuracy_value
    );

    let splitratio = train_test_split(&valueinput.0, &valueinput.1, 0.2, true, Some(2811));
    let randomtest = KNNClassifier::fit(&splitratio.0, &splitratio.2, Default::default()).unwrap();
    let randompredict = randomtest.predict(&splitratio.1).unwrap();
    let accuracytestsplit = accuracy(&splitratio.3, &randompredict);
    println!(
        "The  accuracy of the predicted values on the split basis is: {}",
        accuracytestsplit
    );

    let splitpredict = randomtest.predict(&predvalue).unwrap();
    let mut filewrite = File::create("splitpredict.txt").expect("file not present");
    for i in splitpredict.iter() {
        writeln!(filewrite, "{}", i).expect("line not present");
    }

    Ok("The KNN classifier has been completed".to_string())
}
