use csv::Reader;
use itertools::Itertools;
use linregress::{FormulaRegressionBuilder, RegressionDataBuilder};
use std::borrow::Cow;
use std::error::Error;
use std::fs::File;

use std::process::Command;

fn main() -> Result<(), Box<dyn Error>> {
    // limpiar
     Command::new("cmd")
     .args(&["/C", "cls"])
     .status()
     .expect("Error al ejecutar el comando cls");
    println!("\n\n BUAP 2025   Enrique R.P. Buendia Lozada");

    // Leer el archivo CSV
    let file = File::open("ndatos.csv")?;
    let mut rdr = Reader::from_reader(file);
    
    // Obtener los encabezados de las columnas
    let headers = rdr.headers()?.clone();
    let column_names: Vec<String> = headers.iter().map(|s| s.to_string()).collect();
    
    // Verificar que haya al menos una columna predictora
    if column_names.len() < 2 {
        return Err("El archivo CSV debe tener al menos dos columnas (predictoras y objetivo)".into());
    }
    
    // La última columna es la variable objetivo (fc_max)
    let target = column_names.last().unwrap();
    let predictors: Vec<String> = column_names[..column_names.len()-1].to_vec();
    
    // Preparar almacenamiento de datos
    let mut data: Vec<Vec<f64>> = vec![Vec::new(); column_names.len()];
    
    // Leer todos los registros
    for result in rdr.records() {
        let record = result?;
        for (i, field) in record.iter().enumerate() {
            data[i].push(field.parse::<f64>()?);
        }
    }
    
    // Generar todas las combinaciones posibles de predictores
    let mut best_model = None;
    let mut best_rsquared = f64::NEG_INFINITY;
    
    // Probar todas las combinaciones de 1 a n predictores
    for k in 1..=predictors.len() {
        for combo in predictors.iter().combinations(k) {
            let formula = format!("{} ~ {}", target, combo.iter().join(" + "));
            
            // Construir datos de regresión
            let mut regression_data = vec![(target.as_str(), data[column_names.len()-1].clone())];
            
            for predictor in &combo {
                if let Some(idx) = predictors.iter().position(|x| x == *predictor) {
                    regression_data.push((predictor.as_str(), data[idx].clone()));
                }
            }
            
            let regression_data = RegressionDataBuilder::new()
                .build_from(regression_data)?;
            
            // Ajustar el modelo
            if let Ok(model) = FormulaRegressionBuilder::new()
                .data(&regression_data)
                .formula(&formula)
                .fit() 
            {
                println!("\nModelo: {}", formula);
                println!("R²: {:.4}", model.rsquared);
                
                // Mostrar la ecuación
                let intercept = model.parameters.intercept_value;
                print!("Ecuación: {} = {:.2}", target, intercept);
                
                for (i, var) in combo.iter().enumerate() {
                    print!(" + {:.2}*{}", model.parameters.regressor_values[i], var);
                }
                println!();
                
                // Seguir el mejor modelo
                if model.rsquared > best_rsquared {
                    best_rsquared = model.rsquared;
                    best_model = Some((formula, model));
                }
            }
        }
    }
    
    // Mostrar el mejor modelo
    if let Some((best_formula, best)) = best_model {
        println!("\n\nMEJOR MODELO: {}", best_formula);
        println!("R²: {:.4}", best.rsquared);
        
        let intercept = best.parameters.intercept_value;
        print!("Ecuación: {} = {:.2}", target, intercept);
        
        let vars: Vec<&str> = best_formula.split('~').nth(1).unwrap()
            .split('+')
            .map(|s| s.trim())
            .collect();
        
        for (i, var) in vars.iter().enumerate() {
            print!(" + {:.2}*{}", best.parameters.regressor_values[i], var);
        }
        println!();
    } else {
        println!("No se pudo ajustar ningún modelo válido");
    }
    
     // Mantener la consola abierta después de salir
    Command::new("cmd")
            .args(&["/C", "cmd /k"])
            .status()
            .expect("Error al ejecutar el comando cmd /k");



    Ok(())
}