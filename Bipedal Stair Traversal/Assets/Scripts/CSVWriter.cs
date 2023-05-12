using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class CSVWriter : MonoBehaviour
{
    private string fileName = "data.csv";
    private string filePath;
    private string delimiter = ",";
    private StreamWriter fileWriter;


    public void initialiseJointDataLogging() {
        fileName =  System.DateTime.Now.ToString().Replace("/", "-").Replace(":", "-") + ".csv";

        string desktopPath = System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop);
        filePath = Path.Combine(desktopPath + "/Blind-Bipedal-Locomotion/PCALog/", fileName);
        fileWriter = new StreamWriter(filePath, true);
        // Write the headers of the CSV file
        //fileWriter.WriteLine("Time, J0, J1, J2, J3, J4, J5, J6, J7, J8, J9");
        fileWriter.Flush();
    }

    public void initialiseNeuronDataLogging() {
        fileName =  System.DateTime.Now.ToString().Replace("/", "-").Replace(":", "-") + ".csv";

        string desktopPath = System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop);
        filePath = Path.Combine(desktopPath + "/Blind-Bipedal-Locomotion/NeuronLog/", fileName);
        fileWriter = new StreamWriter(filePath, true);
    }

    public void WriteData(float[] data, bool logTimeFlag) {
        if(logTimeFlag) {
            string currentTime = System.DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString();
            fileWriter.Write(currentTime);
        }

        // Write the data to the CSV file
        for (int i = 0; i < data.Length; i++) {
            fileWriter.Write(data[i] + delimiter);
        }
        fileWriter.WriteLine("");
        fileWriter.Flush();
    }

    /// <summary>
    /// Close the CSV file when the application is quit
    /// </summary>
    private void OnApplicationQuit() {
        fileWriter.Close();
    }
}