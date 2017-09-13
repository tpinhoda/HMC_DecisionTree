package filtrarrbf;

import java.io.File;
import java.io.IOException;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

public class FiltrarRBF {

    public static void main(String[] args) throws IOException, Exception {
        // TODO code application logic here
        String file = args[0];
        Filter replaceMissing = new ReplaceMissingValues();
        //Filter nominalToBinary = new NominalToBinary();
        Filter removeUseless = new RemoveUseless();
        //Filter normalize = new Normalize();
       // Filter standardize = new Standardize();

        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(file));
        Instances train = loader.getDataSet();
        train.setClassIndex(train.numAttributes() - 2);
        replaceMissing.setInputFormat(train);
        train = Filter.useFilter(train, replaceMissing);
        //nominalToBinary.setInputFormat(train);
        //train = Filter.useFilter(train, nominalToBinary);
        removeUseless.setInputFormat(train);
        train = Filter.useFilter(train, removeUseless);
        train.setClassIndex(train.numAttributes() - 1); //GAMBIARRA PRO ID

//        normalize.setInputFormat(train);
//        train = Filter.useFilter(train, normalize);
        //standardize.setInputFormat(train);
       // train = Filter.useFilter(train, standardize);
        
        CSVSaver saver = new CSVSaver();
        saver.setInstances(train);
        saver.setFile(new File(file.replace(".csv", "_fixed.csv")));
        saver.setDestination(new File(file.replace(".csv", "_fixed.csv")));
        saver.writeBatch();

    }

}
