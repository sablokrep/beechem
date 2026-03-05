use ndarray::Array1;
use rdkit::{Properties, ROMol};
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
codeprog@icloud.com
 */

pub fn readsmiles(
    pathfile: &str,
    sequencefile: &str,
    genexpression: &str,
    threshold: &str,
) -> Result<(DenseMatrix<f64>, Vec<i32>), Box<dyn Error>> {
    let fileopen = File::open(pathfile).expect("file not present");
    let fileread = BufReader::new(fileopen);
    let mut smilesvec: Vec<String> = Vec::new();
    for i in fileread.lines() {
        let line = i.expect("file not present");
        let valueadd = line.split(",").collect::<Vec<_>>();
        smilesvec.push(valueadd[4].to_string());
    }
    let mut valuesmiles: Vec<Vec<f64>> = Vec::new();
    for i in smilesvec.iter() {
        let smileparse = ROMol::from_smiles(i).unwrap();
        let properties: HashMap<String, f64> = Properties::new().compute_properties(&smileparse);
        let mut values: Vec<f64> = Vec::new();
        for (_, val) in properties.iter() {
            values.push(*val);
        }
        valuesmiles.push(values);
    }

    let valueexpression = expression(genexpression).unwrap();
    let unifiedseq = sequenceseq(sequencefile).unwrap();

    let mut classseq: Vec<i32> = Vec::new();
    for i in valueexpression.iter() {
        if i.to_string().parse::<i32>().unwrap() < threshold.to_string().parse::<i32>().unwrap() {
            classseq.push(0)
        } else if i.to_string().parse::<i32>().unwrap()
            > threshold.to_string().parse::<i32>().unwrap()
        {
            classseq.push(1)
        }
    }

    let mut combinedvecfinal: Vec<Vec<f64>> = Vec::new();

    for i in valueexpression.iter() {
        for smile in valuesmiles.iter() {
            for seq in unifiedseq.iter() {
                let veca = Array1::from_vec(smile.clone());
                let vecb = Array1::from_vec(seq.clone());
                let combinedvec = veca + vecb;
                let mut combinealter = combinedvec.to_vec();
                combinealter.push(*i);
                combinedvecfinal.push(combinealter);
            }
        }
    }

    let finaldesnsematrix = DenseMatrix::from_2d_vec(&combinedvecfinal).unwrap();
    Ok((finaldesnsematrix, classseq))
}

pub fn expression(pathfile: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let fileopen = File::open(pathfile).expect("file not present");
    let fileread = BufReader::new(fileopen);
    let mut expressionvec: Vec<f64> = Vec::new();
    for i in fileread.lines() {
        let value = i.expect("file not present");
        expressionvec.push(value.parse::<f64>().unwrap());
    }
    Ok(expressionvec)
}

pub fn sequenceseq(pathfile: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let filepath = File::open(pathfile).expect("file not present");
    let fileread = BufReader::new(filepath);
    let mut stringvec = Vec::new();
    for i in fileread.lines() {
        let line = i.expect("file not present");
        if !line.starts_with(">") {
            stringvec.push(line);
        }
    }
    let mut seqvec: Vec<Vec<f64>> = Vec::new();
    for i in stringvec.iter() {
        let charsval = i.chars().collect::<Vec<_>>();
        let mut charseq: Vec<Vec<f64>> = Vec::new();
        for i in charsval.iter() {
            match i {
                'A' => charseq.push(vec![1.0, 0.0, 0.0, 0.0]),
                'T' => charseq.push(vec![0.0, 1.0, 0.0, 0.0]),
                'G' => charseq.push(vec![0.0, 0.0, 1.0, 0.0]),
                'C' => charseq.push(vec![0.0, 0.0, 0.0, 1.0]),
                _ => continue,
            }
        }
        seqvec.push(charseq.iter().cloned().flatten().collect::<Vec<f64>>());
    }
    Ok(seqvec)
}

pub fn readsmiles_predict(
    pathfile: &str,
    sequencefile: &str,
    genexpression: &str,
) -> Result<DenseMatrix<f64>, Box<dyn Error>> {
    let fileopen = File::open(pathfile).expect("file not present");
    let fileread = BufReader::new(fileopen);
    let mut smilesvec: Vec<String> = Vec::new();
    for i in fileread.lines() {
        let line = i.expect("file not present");
        let valueadd = line.split(",").collect::<Vec<_>>();
        smilesvec.push(valueadd[4].to_string());
    }
    let mut valuesmiles: Vec<Vec<f64>> = Vec::new();
    for i in smilesvec.iter() {
        let smileparse = ROMol::from_smiles(i).unwrap();
        let properties: HashMap<String, f64> = Properties::new().compute_properties(&smileparse);
        let mut values: Vec<f64> = Vec::new();
        for (_, val) in properties.iter() {
            values.push(*val);
        }
        valuesmiles.push(values);
    }

    let valueexpression = expression(genexpression).unwrap();
    let unifiedseq = sequenceseq(sequencefile).unwrap();
    let mut combinedvecfinal: Vec<Vec<f64>> = Vec::new();

    for i in valueexpression.iter() {
        for smile in valuesmiles.iter() {
            for seq in unifiedseq.iter() {
                let veca = Array1::from_vec(smile.clone());
                let vecb = Array1::from_vec(seq.clone());
                let combinedvec = veca + vecb;
                let mut combinealter = combinedvec.to_vec();
                combinealter.push(*i);
                combinedvecfinal.push(combinealter);
            }
        }
    }

    let finaldesnsematrix = DenseMatrix::from_2d_vec(&combinedvecfinal).unwrap();
    Ok(finaldesnsematrix)
}
