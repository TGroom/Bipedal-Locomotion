using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using BodyPart = Unity.MLAgentsExamples.BodyPart;
using Random = UnityEngine.Random;
using VonMisesDistribution = Accord.Statistics.Distributions.Univariate.VonMisesDistribution;

using System.Linq;
using System.Collections.Generic;

public class BipedalAgent : Agent {

    public bool printEnable = false;  // Flag to enable debugging variables using Unity grapher
    public bool jointLogEnable = false;  // Flag to enable logging of joint angles to a csv file

    public float TargetWalkingSpeed = 1.0f;  // The target walking speed
    const float m_maxWalkingSpeed = 1.5f;  // The max walking speed when using random walking speed each episode
    public bool randomizeWalkSpeedEachEpisode;  // Flag to enable randomising walking speed each episode

    public float walkCycleTime = 0.7f;  // Time in seconds for a single walking gait (used for phase input of ANN and reward caluclation)

    const float Amu = -0.5f * Mathf.PI;  // When during the phase should the control switch legs (in range -pi to pi)
    const float Bmu = 0.5f * Mathf.PI;  // When during the phase should the control switch back to original leg (in range -pi to pi)
    const float kappa = 25;  // Sets the harshness of the transition from one leg to the other in the cycle !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! EXPLAINE BETTER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    VonMisesDistribution vonMisesCDF_A = new VonMisesDistribution(Amu, kappa);  // Create a vonMises CDF object with parameters Amu (mean) and kappa (variance)
    VonMisesDistribution vonMisesCDF_B = new VonMisesDistribution(Bmu, kappa);  // Create a vonMises CDF object with parameters Bmu (mean) and kappa (variance)

    // List storing the multipliers for each reward component when calculating total reward
    private float[] rewardWeights =  {0.2f, 0.2f, 0.15f, 0.15f, 0.15f, 0.15f}; //{0.14f, 0.14f, 0.14f, 0.14f, 0.14f, 0.14f, 0.14f, 0.0273333f, 0.0273333f, 0.0273333f};

    // The target heading for the agent to match
    private Quaternion targetRotation = Quaternion.Euler(0f, 0f, 0f);  // Set default value

    [Header("Body Parts")]
    public Transform body;
    public Transform hipLeft;
    public Transform hipRight;
    public Transform upperLegLeft;
    public Transform upperLegRight;
    public Transform lowerLegLeft;
    public Transform lowerLegRight;
    
    //public Transform footLeft;
    //public Transform footRight;

    [Header("Scene")]
    public GameObject ground;
    public bool groundRandEnabled = true;
    public GameObject step;  // The step or stair object
    GameObject[] steps = new GameObject[10];
    public bool stairsEnabled = false;
    public bool stepEnabled = false;
    public bool externalForces = false;
    public float perturbationScale = 600f;
    public bool externalBalls = false;

    public GameObject obstacle;  // The step or stair object
    public float repeatInterval = 1.0f; // Repeat interval in seconds
    private float nextInvokeTime = 0.0f;

    private CSVWriter csv;  

    // Because ragdolls can move erratically during training, using a stabilized reference transform improves learning !!!!!!!!!!!!!!! CHANGE THIS??? !!!!!!!!!!!!!!!!!!
    OrientationCubeController m_OrientationCube;
    JointDriveController m_JdController;

    private float[] previousContinuousActions = new float[18];
    private float at = 0;  // Magnitude of the total change in action
    private Vector3 bodyLastVelocity = new Vector3(0, 0, 0);
    private Vector3 bodyLastAngularVelocity = new Vector3(0, 0, 0);
    //private float[][] actionList = new float[200][];
    //private int count = 100;
    //private int oppositeActionIdx = 0;

    public bool noiseEnable = false;
    public float noisePercent = 0.02f;

    public bool runTestRoutine = false;
    private float episodeBeginTime;
    private int episodeCount = 0;

    private List<float> footLstart = new List<float>();
    private List<float> footRstart = new List<float>();
    private List<float> footLstop = new List<float>();
    private List<float> footRstop = new List<float>();
    private List<float> epLength = new List<float>();
    private List<int> groundContactListL = new List<int>();
    private List<int> groundContactListR = new List<int>();

    private bool oldTouchingGroundL;
    private bool oldTouchingGroundR;

    private float timeOfLeftContact = 0;


    private float sumReward = 0;
    private int noOfRewardsLogged = 0;
    private float averageReward = 0;

    private Vector3 lastPosL = new Vector3(0, 0, 0);
    private Vector3 lastPosR = new Vector3(0, 0, 0);

    private List<float> stabilityMetric = new List<float>();

    private List<float> localFootLPosListX =  new List<float>();
    private List<float> localFootLPosListZ =  new List<float>();
    private List<float> localFootRPosListX =  new List<float>();
    private List<float> localFootRPosListZ =  new List<float>();

    private List<float[]> jointPosArr = new List<float[]>();

    private List<float>[] rewardListLog = new List<float>[6];

    private float XL = 0;
    private float XR = 0;

    private bool check1 = true;
    private bool check2 = true;
    private bool check3 = true;

    public override void Initialize() {

        if (jointLogEnable) csv = ground.AddComponent<CSVWriter>();  // The script instantiation for the csv logger
        if (jointLogEnable) csv.initialiseJointDataLogging();

        m_OrientationCube = GetComponentInChildren<OrientationCubeController>();
        m_JdController = GetComponent<JointDriveController>();
        m_JdController.SetupBodyPart(body);
        m_JdController.SetupBodyPart(hipLeft);
        m_JdController.SetupBodyPart(hipRight);
        m_JdController.SetupBodyPart(upperLegLeft);
        m_JdController.SetupBodyPart(upperLegRight);
        m_JdController.SetupBodyPart(lowerLegLeft);
        m_JdController.SetupBodyPart(lowerLegRight);
        
        //m_JdController.SetupBodyPart(footLeft);
        //m_JdController.SetupBodyPart(footRight);

        if (stairsEnabled || stepEnabled) {
            // Instansiate 10 steps
            steps[0] = step;
            for (int i = 1; i < 10; i++) {
                steps[i] = Instantiate(step, step.transform);
                steps[i].transform.SetParent(ground.transform);
                //steps[i].transform.parent = gameObject.transform;
            }
        }

        RandomiseDomain();
        rewardListLog[0] = new List<float>();
        rewardListLog[1] = new List<float>();
        rewardListLog[2] = new List<float>();
        rewardListLog[3] = new List<float>();
        rewardListLog[4] = new List<float>();
        rewardListLog[5] = new List<float>();
    }


    public override void OnEpisodeBegin() {

        // Reset the orientation and position of the agent and its body parts
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values) {
            bodyPart.Reset(bodyPart);
        }

        body.rotation = Quaternion.Euler(-90f, 0f, 0f);
        targetRotation = Quaternion.Euler(0f, 0f, 0f);

        m_OrientationCube.transform.rotation = targetRotation;  // Update orientation object

        TargetWalkingSpeed = randomizeWalkSpeedEachEpisode ? Random.Range(0.5f, m_maxWalkingSpeed) : TargetWalkingSpeed;

        RandomiseDomain();

        // Set joint strength to maximum torque of 100Nm
        m_JdController.bodyPartsDict[hipLeft].SetJointStrength(300);
        m_JdController.bodyPartsDict[hipRight].SetJointStrength(300);
        m_JdController.bodyPartsDict[upperLegLeft].SetJointStrength(300);
        m_JdController.bodyPartsDict[upperLegRight].SetJointStrength(300);
        m_JdController.bodyPartsDict[lowerLegLeft].SetJointStrength(300);
        m_JdController.bodyPartsDict[lowerLegRight].SetJointStrength(300);
        
        //m_JdController.bodyPartsDict[footRight].SetJointStrength(100*4);
        //m_JdController.bodyPartsDict[footLeft].SetJointStrength(100*4);
        
        //actionList = new float[100][];
        //count = 100;
        //oppositeActionIdx = 0;

        if (runTestRoutine) {
            if(episodeCount > 0){
                epLength.Add(Time.fixedTime - episodeBeginTime);
  
                Debug.Log(episodeCount);

                if (episodeCount % 1 == 0){ //------------------------------------------------------
                    var s = "";
                
                    s += "Rup: " + footRstop.Average();
                    s += ", Rdown: " + footRstart.Average();
                    s += ", Lup: " + footLstop.Average();
                    s += ", footLdown: " + footLstart.Average();

                
                    s += ", RupSD: " + standardDeviation(footRstop);
                    s += ", RdownSD: " + standardDeviation(footRstart);
                    s += ", LupSD: " + standardDeviation(footLstop);
                    s += ", footLstartSD: " + standardDeviation(footLstart);

                    s += ", epLengthAverage: " + epLength.Average();
                    s += ", epLengthSD: " + standardDeviation(epLength);

                    s += ", stabilityMetricAverageCoPfromCoM: " + stabilityMetric.Average();

                    float ASI = (2 * Mathf.Abs(XR - XL) / Mathf.Abs(XR + XL)) * 100;  // Calculate Actuation Symmetry Index
                    s += ", ASI: " + ASI;

                    s += "\nReward 1: " + rewardListLog[0].Average();
                    s += "\nReward 2: " + rewardListLog[1].Average();
                    s += "\nReward 3: " + rewardListLog[2].Average();
                    s += "\nReward 4: " + rewardListLog[3].Average();
                    s += "\nReward 5: " + rewardListLog[4].Average();
                    s += "\nReward 6: " + rewardListLog[5].Average();

                    //for (int i = 0; i < MaxStep - 1; i++) {
                    //    s += groundContactListL[i] + ", " + groundContactListR[i] + "\n";
                    //}
                    Debug.Log(s);
                    //s = "";
                    //for(int i = 0; i < (MaxStep/5)-1; i++) {
                    //    //s += (localFootLPosListZ[i] + localFootLPosListZ[i+(int)(0.885f*200)])/2 + ", " 
                    //    //    + -(localFootLPosListX[i] + localFootLPosListX[i+(int)(0.885f*200)])/2 + ", " 
                   //     //    + (localFootRPosListZ[i] + localFootRPosListZ[i+(int)(0.885f*200)])/2 + ", " 
                   //     //    + -(localFootRPosListX[i] + localFootRPosListX[i+(int)(0.885f*200)])/2 + "\n" ;

                        //s += jointPosArr[i][0] + ", " + jointPosArr[i][1] + ", " + jointPosArr[i][2] + ", " + jointPosArr[i][3] + ", " + jointPosArr[i][4] + ", " + jointPosArr[i][5] + "\n";
                    //}
                    
                    //Debug.Log(s);
                }
            }
            
            episodeBeginTime = Time.fixedTime;
            episodeCount++;
        }

        sumReward = 0;
        noOfRewardsLogged = 0;
        averageReward = 0;

        if(printEnable)
        {
            lastPosL = new Vector3(0, 0, 0);
            lastPosR = new Vector3(0, 0, 0);
        }

        check1 = true;
        check2 = true;
        check3 = true;
    }

    /// <summary>
    /// Update the sensor observations from a body parts
    /// </summary>
    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor) {
        //if(bp.rb.transform != hipLeft || bp.rb.transform != hipRight || bp.rb.transform != upperLegLeft || bp.rb.transform != upperLegRight || bp.rb.transform != lowerLegLeft || bp.rb.transform != lowerLegRight) { // Not adding sensors to the hip
        
        // Is this bp touching the ground
        sensor.AddObservation(bp.groundContact.touchingGround); 

        // Get velocities in orientation cube's space
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.velocity) * noise());
        //sensor.AddObservation(phaseBasedMirroring(m_OrientationCube.transform.InverseTransformDirection(bp.rb.angularVelocity)));  // More realistic???

        // Get position relative to body in orientation cube's space
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.position - body.position) * noise());

        if (bp.rb.transform != body) {
            sensor.AddObservation(bp.rb.transform.localRotation);
            // sensor.AddObservation(phaseBasedMirroring(bp.currentStrength / m_JdController.maxJointForceLimit));
            sensor.AddObservation(bp.joint.targetPosition * noise());
        }
    }

    /// <returns>
    /// The current time in the phase from 0 to 2pi in the interval of walkCycleTime
    /// </returns>
    private float getPhase() {
        return ((Time.fixedTime / walkCycleTime) * (2f * Mathf.PI)) % (2f * Mathf.PI);  // e.g. 0 seconds = 0, 0.7 seconds = 2pi, etc
    }
    
    //private float phaseBasedMirroring(float value) {
    //    return (getPhase() < Mathf.PI) ? -value : value;
    //}


   // private Vector3 phaseBasedMirroring(Vector3 value) {   
    //    return (getPhase() < Mathf.PI) ? -value : value;
    //}

    //private Quaternion phaseBasedMirroring(Quaternion value) {
   //     return (getPhase() < Mathf.PI) ? Quaternion.Inverse(value) : value;
   // }

   // private bool phaseBasedMirroring(bool value) {
  //      return (getPhase() < Mathf.PI) ? !value: value;
  //  }

    private float noise(){
        if (noiseEnable) {
            return 1f + Random.Range(-noisePercent/2, noisePercent/2);
        }
        return 1f;
    }

    /// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor) {
        var cubeForward = m_OrientationCube.transform.forward;

        //velocity to match
        var velGoal = cubeForward * TargetWalkingSpeed;

        var avgVel = GetAvgVelocity();  //ragdoll's avg vel
        sensor.AddObservation(Vector3.Distance(velGoal, avgVel)*noise());  //current ragdoll velocity. normalized
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel)*noise());  //avg body vel relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(velGoal));  //vel goal relative to cube
        sensor.AddObservation(Quaternion.FromToRotation(body.forward, cubeForward));  //rotation deltas

        if (firstHalfOfCycle()) {

            CollectObservationBodyPart(m_JdController.bodyPartsDict[hipLeft], sensor);
            CollectObservationBodyPart(m_JdController.bodyPartsDict[upperLegLeft], sensor);
            CollectObservationBodyPart(m_JdController.bodyPartsDict[lowerLegLeft], sensor);

            CollectObservationBodyPart(m_JdController.bodyPartsDict[hipRight], sensor);
            CollectObservationBodyPart(m_JdController.bodyPartsDict[upperLegRight], sensor);
            CollectObservationBodyPart(m_JdController.bodyPartsDict[lowerLegRight], sensor);

        } else {
            
            CollectObservationBodyPart(m_JdController.bodyPartsDict[hipRight], sensor);
            CollectObservationBodyPart(m_JdController.bodyPartsDict[upperLegRight], sensor);
            CollectObservationBodyPart(m_JdController.bodyPartsDict[lowerLegRight], sensor);

            CollectObservationBodyPart(m_JdController.bodyPartsDict[hipLeft], sensor);
            CollectObservationBodyPart(m_JdController.bodyPartsDict[upperLegLeft], sensor);
            CollectObservationBodyPart(m_JdController.bodyPartsDict[lowerLegLeft], sensor);

        }

        //float clockSensor1 = (Mathf.Sin(getPhase() + (Mathf.PI/2)) + 1f) / 2f;  // from time to value between 0 and 1  //CHANGE TO -1 and 1
        //float clockSensor2 = (Mathf.Sin(getPhase() - (Mathf.PI/2)) + 1f) / 2f;  // from time to value between 0 and 1

        //sensor.AddObservation((clockSensor1));
        //sensor.AddObservation((clockSensor2));
        //sensor.AddObservation(getPhase());
        //sensor.AddObservation((2f * Mathf.PI) - getPhase());


        //if (printEnable) {
        //    Grapher.Log(clockSensor1, "clockSensor1");
        //    Grapher.Log(clockSensor2, "clockSensor2");
           // Grapher.Log(phaseBasedMirroring(1), "conditionalReverse");
        //}
    }


    private bool firstHalfOfCycle()
    {
        return (bool)(getPhase() < Mathf.PI);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers) {
        var bpDict = m_JdController.bodyPartsDict;
        var continuousActions = actionBuffers.ContinuousActions;  // All between -1 and 1
        
        if (firstHalfOfCycle()) {

            // Left
            bpDict[hipLeft].SetJointTargetRotation(continuousActions[0], 0, 0);
            bpDict[upperLegLeft].SetJointTargetRotation(continuousActions[1], 0, 0);
            bpDict[lowerLegLeft].SetJointTargetRotation(continuousActions[2], 0, 0);

            // Right
            bpDict[hipRight].SetJointTargetRotation(continuousActions[3], 0, 0);
            bpDict[upperLegRight].SetJointTargetRotation(continuousActions[4], 0, 0);
            bpDict[lowerLegRight].SetJointTargetRotation(continuousActions[5], 0, 0);
            //if (printEnable) Debug.Log("1");
        } else
        {
            // Left
            bpDict[hipLeft].SetJointTargetRotation(continuousActions[3], 0, 0);
            bpDict[upperLegLeft].SetJointTargetRotation(continuousActions[4], 0, 0);
            bpDict[lowerLegLeft].SetJointTargetRotation(continuousActions[5], 0, 0);

            // Right
            bpDict[hipRight].SetJointTargetRotation(continuousActions[0], 0, 0);
            bpDict[upperLegRight].SetJointTargetRotation(continuousActions[1], 0, 0);
            bpDict[lowerLegRight].SetJointTargetRotation(continuousActions[2], 0, 0);
            //if (printEnable) Debug.Log("2");
        }

        
        
        //bpDict[footRight].SetJointTargetRotation(phaseBasedMirroring(continuousActions[++i]), phaseBasedMirroring(continuousActions[++i]), 0);
        //bpDict[footLeft].SetJointTargetRotation(phaseBasedMirroring(continuousActions[++i]), phaseBasedMirroring(continuousActions[++i]), 0);
                


        if (runTestRoutine)
        {
            /*
            Vector3 PosLlocal = PosL - new Vector3(0, 0, CoM.z);
            //Debug.DrawRay(PosLlocal, Vector3.up, Color.black);
            localFootLPosListX.Add(PosLlocal.x - lowerLegLeft.localPosition.x);
            localFootLPosListZ.Add(PosLlocal.z - lowerLegLeft.localPosition.z);

            Vector3 PosRlocal = PosR - new Vector3(0, 0, CoM.z);
            //Debug.DrawRay(PosLlocal, Vector3.up, Color.black);
            localFootRPosListX.Add(PosRlocal.x - lowerLegRight.localPosition.x);
            localFootRPosListZ.Add(PosRlocal.z - lowerLegRight.localPosition.z);
            */
            float[] jointPosList = {0,0,0,0,0,0};

            jointPosList[0] = m_JdController.bodyPartsDict[hipLeft].rb.transform.localRotation.eulerAngles.x;
            jointPosList[1] = m_JdController.bodyPartsDict[hipRight].rb.transform.localRotation.eulerAngles.x;
            jointPosList[2] = m_JdController.bodyPartsDict[upperLegLeft].rb.transform.localRotation.eulerAngles.x;
            jointPosList[3] = m_JdController.bodyPartsDict[upperLegRight].rb.transform.localRotation.eulerAngles.x;
            jointPosList[4] = m_JdController.bodyPartsDict[lowerLegLeft].rb.transform.localRotation.eulerAngles.x;
            jointPosList[5] = m_JdController.bodyPartsDict[lowerLegRight].rb.transform.localRotation.eulerAngles.x;
            jointPosArr.Add(jointPosList);
        }

        at = 0;
        int num = 0;
        foreach (var action in continuousActions) {  // Calculated the magnitude of the action vector
            at += Mathf.Pow((float)action - previousContinuousActions[num], 2);
            previousContinuousActions[num] = (float)action;
            num++;
        }
        at = Mathf.Sqrt(at);


    }


    void FixedUpdate() {

        m_OrientationCube.transform.rotation = targetRotation;  // Update orientation object
        var cubeForward = m_OrientationCube.transform.forward;

        //This reward will approach 1 if it matches perfectly and approach zero as it deviates
        //var matchSpeedReward = GetMatchingVelocityReward(cubeForward * TargetWalkingSpeed, GetAvgVelocity());
        Vector3 desiredVel = cubeForward * TargetWalkingSpeed;
        Vector3 actualVel = GetAvgVelocity();



        float pA = (float)vonMisesCDF_A.DistributionFunction(getPhase() - Mathf.PI);  // Input must be -pi to pi
        float pB = (float)vonMisesCDF_B.DistributionFunction(getPhase() - Mathf.PI);
        float EI = pA * (1 - pB); // pA and pB

        m_JdController.GetCurrentJointForces();
        float totalTorque = 0;
        
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values) {
            totalTorque += bodyPart.currentJointTorqueSqrMag;
            
        }

        if(StepCount > 400 && StepCount < MaxStep - 200 && runTestRoutine) {
            float temp = 0;
            temp += Mathf.Pow(m_JdController.bodyPartsDict[hipLeft].currentJointTorqueSqrMag, 2);
            temp += Mathf.Pow(m_JdController.bodyPartsDict[upperLegLeft].currentJointTorqueSqrMag, 2);
            temp += Mathf.Pow(m_JdController.bodyPartsDict[lowerLegLeft].currentJointTorqueSqrMag, 2);
            XL += Mathf.Sqrt(temp);
        
            temp = 0;
            temp += Mathf.Pow(m_JdController.bodyPartsDict[hipRight].currentJointTorqueSqrMag, 2);
            temp += Mathf.Pow(m_JdController.bodyPartsDict[upperLegRight].currentJointTorqueSqrMag, 2);
            temp += Mathf.Pow(m_JdController.bodyPartsDict[lowerLegRight].currentJointTorqueSqrMag, 2);
            XR += Mathf.Sqrt(temp);

            // Add calulcation of SI?
        }

        Vector3 lowerLegLeftJointForces = m_JdController.bodyPartsDict[lowerLegLeft].currentJointForce;
        Vector3 lowerLegRightJointForces = m_JdController.bodyPartsDict[lowerLegRight].currentJointForce;
        Vector3 lowerLegLeftGravityForce =  m_JdController.bodyPartsDict[lowerLegLeft].rb.mass * Physics.gravity;  // Calculate the force due to gravity on the body
        Vector3 lowerLegRightGravityForce = m_JdController.bodyPartsDict[lowerLegRight].rb.mass * Physics.gravity;  // Calculate the force due to gravity on the body
        Vector3 lowerLegLeftReactionForce = Convert.ToInt32(m_JdController.bodyPartsDict[lowerLegLeft].rb.GetComponent<GroundContact>().touchingGround) * -(lowerLegLeftGravityForce + lowerLegLeftJointForces);
        Vector3 lowerLegRightReactionForce = Convert.ToInt32(m_JdController.bodyPartsDict[lowerLegRight].rb.GetComponent<GroundContact>().touchingGround) * -(lowerLegRightGravityForce + lowerLegRightJointForces);
 
        if (lowerLegLeftReactionForce.y < 0) lowerLegLeftReactionForce = new Vector3(0,0,0);
        if (lowerLegRightReactionForce.y < 0) lowerLegRightReactionForce = new Vector3(0,0,0);

        Vector3 CoM = m_JdController.bodyPartsDict[body].rb.worldCenterOfMass;

        // Account for transform due to COG of lower leg and contact point being different
       // Vector3 PosL = m_JdController.bodyPartsDict[lowerLegLeft].rb.ClosestPointOnBounds(new Vector3(0,-10000,0));
       // Vector3 PosR = m_JdController.bodyPartsDict[lowerLegRight].rb.ClosestPointOnBounds(new Vector3(0,-10000,0));
        Vector3 PosL = (lowerLegLeft.position - lowerLegLeft.forward * 0.9f) - lowerLegLeft.right * 0.1f;
        Vector3 PosR = (lowerLegRight.position - lowerLegRight.forward * 0.9f) - lowerLegRight.right * 0.1f;
        

        
        //footLPos.Add(lowerLegLeft.localPosition - lowerLegLeft.forward * 0.9f) - lowerLegLeft.right * 0.1f;);
        //footRPos.Add(lowerLegRight.localPosition);

        PosL.y = 0;
        PosR.y = 0;
        //PosL.y = 0;
        //PosR.y = 0;
        float FyL = lowerLegLeftReactionForce.y;
        float FyR = lowerLegRightReactionForce.y;
        float deltaX = (-PosL.x + PosR.x);
        float deltaZ = (-PosL.z + PosR.z);


        //float Mx, My, Mz;  // Moments around x, y, and z axes, respectively

        float CoP_x, CoP_z; // Center of Pressure coordinates
        //float ZMP_x, ZMP_z; // Zero Moment Point coordinates

        float Fy = FyL + FyR;

        //Vector3 bodyAngularAcc = (m_JdController.bodyPartsDict[body].rb.angularVelocity - bodyLastAngularVelocity) / Time.fixedDeltaTime;
        //bodyLastAngularVelocity = m_JdController.bodyPartsDict[body].rb.angularVelocity;
        //float MoI =  0.5f * m_JdController.bodyPartsDict[body].rb.mass * (float)Math.Pow(0.1f, 2f); // Moment of inertia modeled as cylinder.
        //Debug.Log(MoI);
        //Mx = bodyAngularAcc.x * MoI;
        //Mz = bodyAngularAcc.y * MoI;
        

        if (Fy > 0) {
            CoP_x = (deltaX * FyR) / Fy;  // relative to left foot
            CoP_z = (deltaZ * FyR) / Fy;
            Vector3 CoP = PosL + new Vector3(CoP_x, 0, CoP_z);  // Center of pressure relative to CoM
            Vector3 projectedCoM = new Vector3(CoM.x, 0, CoM.z);
            stabilityMetric.Add(Vector3.Distance(projectedCoM, CoP));
        }
        /*
        if (Math.Abs(Mz) > 0.0001f && Math.Abs(Mx) > 0.0001f) {
            ZMP_x = (Mx * CoP_z - Mz * CoP_x) / (Mz + Mx);
            ZMP_z = (Mx * CoP_z + Mz * CoP_x) / (Mz + Mx);
        } else if (Math.Abs(Mz) > 0.0001f) {
            ZMP_x = CoP_x - Mz / Fy * CoP_z;
            ZMP_z = 0;
        } else if (Math.Abs(Mx) > 0.0001f) {
            ZMP_x = 0;
            ZMP_z = CoP_z + Mx / Fy * CoP_x;
        } else {
            ZMP_x = CoP_x;
            ZMP_z = CoP_z;
        }

        */

        

        
        //Vector3 ZMP = new Vector3(ZMP_x, 0, ZMP_z);
        

        //Debug.DrawRay(PosL, new Vector3(0, -lowerLegLeftGravityForce.y * 0.01f, 0), Color.green);
        //Debug.DrawRay(PosR, new Vector3(0, -lowerLegRightGravityForce.y * 0.01f, 0), Color.green);
        if(printEnable) {
            //Debug.DrawRay(PosL, lowerLegLeftReactionForce*0.0001f, Color.red);
            //Debug.DrawRay(PosR, lowerLegRightReactionForce*0.0001f, Color.red);
            //Debug.DrawLine(CoM, CoP, Color.blue);
            //Debug.DrawRay(PosL + ZMP, Vector3.up * 0.5f, Color.red);
            Debug.DrawRay(new Vector3(0,0,0), Vector3.up * 0.2f, Color.green);
            Debug.DrawRay(new Vector3(0,0,0), Vector3.forward * 0.2f, Color.blue);
            Debug.DrawRay(new Vector3(0,0,0), Vector3.right * 0.2f, Color.red);

            Debug.DrawRay(lastPosL, PosL - lastPosL, Color.red, 1f, false);
            lastPosL = PosL;
            Debug.DrawRay(lastPosR, PosR - lastPosR, Color.blue, 1f, false);
            lastPosR = PosR;

        }

        if (jointLogEnable) {
            float[] jointArr = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            
            jointArr[0] = m_JdController.bodyPartsDict[hipLeft].rb.transform.localRotation.eulerAngles.x;
            jointArr[1] = m_JdController.bodyPartsDict[hipRight].rb.transform.localRotation.eulerAngles.x;
            jointArr[2] = m_JdController.bodyPartsDict[upperLegLeft].rb.transform.localRotation.eulerAngles.x;
            jointArr[3] = m_JdController.bodyPartsDict[upperLegRight].rb.transform.localRotation.eulerAngles.x;
            jointArr[4] = m_JdController.bodyPartsDict[lowerLegLeft].rb.transform.localRotation.eulerAngles.x;
            jointArr[5] = m_JdController.bodyPartsDict[lowerLegRight].rb.transform.localRotation.eulerAngles.x;
            
            if (jointArr[0] > 180f) jointArr[0] = jointArr[0] - 360f;
            if (jointArr[1] > 180f) jointArr[1] = jointArr[1] - 360f;
            if (jointArr[2] > 180f) jointArr[2] = jointArr[2] - 360f;
            if (jointArr[3] > 180f) jointArr[3] = jointArr[3] - 360f;
            if (jointArr[4] > 180f) jointArr[4] = jointArr[4] - 360f;
            if (jointArr[5] > 180f) jointArr[5] = jointArr[5] - 360f;

            jointArr[6] = m_JdController.bodyPartsDict[lowerLegLeft].groundContact.touchingGround ? 1f : 0f;
            jointArr[7] = m_JdController.bodyPartsDict[lowerLegRight].groundContact.touchingGround ? 1f : 0f;
            jointArr[8] = m_JdController.bodyPartsDict[body].rb.position.y;
            jointArr[9] = m_JdController.bodyPartsDict[body].rb.position.z;

            //jointArr[6] = m_JdController.bodyPartsDict[footRight].rb.transform.localRotation.eulerAngles.x;
            //jointArr[7] = m_JdController.bodyPartsDict[footRight].rb.transform.localRotation.eulerAngles.y;
            //jointArr[8] = m_JdController.bodyPartsDict[footLeft].rb.transform.localRotation.eulerAngles.x;
            //jointArr[9] = m_JdController.bodyPartsDict[footLeft].rb.transform.localRotation.eulerAngles.y;

            csv.WriteData(jointArr, false);
        }

        Vector3 bodyAcceleration = (m_JdController.bodyPartsDict[body].rb.velocity - bodyLastVelocity) / Time.fixedDeltaTime;
        bodyLastVelocity = m_JdController.bodyPartsDict[body].rb.velocity;


        //float leftHipLegAngle = m_JdController.bodyPartsDict[hipLeft].currentEularJointRotation.x;
        //float rightHipLegAngle = m_JdController.bodyPartsDict[hipRight].currentEularJointRotation.x;
        float leftUpperLegAngle = m_JdController.bodyPartsDict[upperLegLeft].currentEularJointRotation.x;
        float rightUpperLegAngle = m_JdController.bodyPartsDict[upperLegRight].currentEularJointRotation.x;
        //float leftLowerLegAngle = m_JdController.bodyPartsDict[lowerLegLeft].rb.transform.localRotation.eulerAngles.x;
        //float rightLowerLegAngle = m_JdController.bodyPartsDict[lowerLegRight].rb.transform.localRotation.eulerAngles.x;
        //var s = leftUpperLegAngle + ", " + rightUpperLegAngle;
        //if(printEnable) Debug.Log(s);
        //float[] jointAngles = new float[] {leftHipLegAngle, rightHipLegAngle, leftUpperLegAngle, rightUpperLegAngle, leftLowerLegAngle, rightLowerLegAngle};

        //count += 1;
        //oppositeActionIdx = (count - (int)((walkCycleTime * 0.5f) / Time.fixedDeltaTime)) % 100;
        //if(printEnable) Grapher.Log(count%27, "count%27");
        //Grapher.Log(oppositeActionIdx, "oppositeActionIdx");
        //actionList[count % 100] = jointAngles;
        //PrintActionSegment(actionList[oppositeActionIdx]);
        //PrintArray(jointAngles);
        //if(printEnable) Grapher.Log(count % 70, "count % 70");
        //if(printEnable) Grapher.Log(actionList[count % 100][0], "jointAnglesNow");
        //if(printEnable) Grapher.Log(actionList[oppositeActionIdx][0], "jointAnglesOld");

        //float sumSym = 0;
        //sumSym += Mathf.Abs(actionList[oppositeActionIdx][0] - actionList[count % 100][1]);
        //sumSym += Mathf.Abs(actionList[oppositeActionIdx][1] - actionList[count % 100][0]);
        //sumSym += Mathf.Abs(actionList[oppositeActionIdx][2] - actionList[count % 100][3]);
        //sumSym += Mathf.Abs(actionList[oppositeActionIdx][3] - actionList[count % 100][2]);
        //sumSym += Mathf.Abs(actionList[oppositeActionIdx][4] - actionList[count % 100][5]);
        //sumSym += Mathf.Abs(actionList[oppositeActionIdx][5] - actionList[count % 100][4]);
        //if(printEnable) Grapher.Log(sumSym, "sumSym");

        //float midPoint = 5f;  // Angle at which the upper legs cross when walking
        //float sensitivity = 0.4f;  // Used to be 0.2f
        //float sensitivity2 = 15f;

        //var leftFoot = m_JdController.bodyPartsDict[footLeft].rb.position;
        //var rightFoot = m_JdController.bodyPartsDict[footRight].rb.position;
        //var s = leftFoot.y - rightFoot.y;
        //if (printEnable) Debug.Log(1 / (1 + Mathf.Exp((((EI * 2) - 1 ) * sensitivity2 * (leftFoot.y - rightFoot.y)) + 5)));
                //This error will approach 0 if it faces the target direction perfectly and approach 1 as it deviates

        if (firstHalfOfCycle()) {

            float[] rewardList = {  //EI * (1 - Mathf.Exp(-0.05f * lowerLegLeftReactionForce.magnitude)),  //(EI / (1 + Mathf.Exp(sensitivity * (leftUpperLegAngle - midPoint)))) + ((1 - EI) / (1 + Mathf.Exp(-sensitivity * (leftUpperLegAngle - midPoint)))),  //(EI * Mathf.Exp(-0.08f * Mathf.Max((leftUpperLegAngle - 330), 0))) + ((1 - EI) * Mathf.Exp(-0.08f * MathF.Abs(Mathf.Min((leftUpperLegAngle - 320), 0)))),  //EI * Mathf.Exp(-0.005f * FyL) 
                                    //(1 - EI) * (1 - Mathf.Exp(-0.05f * lowerLegRightReactionForce.magnitude)), //((1 - EI) / (1 + Mathf.Exp(sensitivity * (rightUpperLegAngle - midPoint)))) + (EI / (1 + Mathf.Exp(-sensitivity * (rightUpperLegAngle - midPoint)))),  //((1 - EI) * Mathf.Exp(-0.08f * Mathf.Max((rightUpperLegAngle - 330), 0))) + (EI * Mathf.Exp(-0.08f * MathF.Abs(Mathf.Min((rightUpperLegAngle - 320), 0)))),  //(1 - EI) * Mathf.Exp(-0.005f * FyR)
                                    //(1 - EI) * (1 - Mathf.Exp(-5f * m_JdController.bodyPartsDict[lowerLegLeft].rb.velocity.magnitude)),  // Should this be angular velocity?
                                    //EI * (1 - Mathf.Exp(-5f * m_JdController.bodyPartsDict[lowerLegRight].rb.velocity.magnitude)),
                                    Mathf.Exp(-2f * (Vector3.Dot(cubeForward, body.up) + 1)),  // lookAtTargetReward
                                    Mathf.Exp(-3f * Mathf.Abs(desiredVel.z - actualVel.z)),  // Forward / Backward Velocity
                                    Mathf.Exp(-20f * Mathf.Abs(actualVel.x)),
                                    Mathf.Exp(-2f * at),
                                    Mathf.Exp(-0.02f * totalTorque), //Used to be -0.004
                                    Mathf.Exp(-0.1f * (m_JdController.bodyPartsDict[body].rb.angularVelocity.magnitude + bodyAcceleration.magnitude))
                                    };

            int num = 0;
            foreach (var item in rewardList) {
                AddReward(rewardWeights[num] * item);
                sumReward += rewardWeights[num] * item;
                num++;
            }

            for (int i = 0; i < 6; i++) {
                rewardListLog[i].Add(rewardList[i]);
            }
            


            noOfRewardsLogged++;
            averageReward = sumReward/noOfRewardsLogged;
            
        } else
        {
            AddReward(averageReward);
            sumReward = 0;
            noOfRewardsLogged = 0;
        }


        
        if (printEnable) {
            Grapher.Log(EI, "EI");
            Grapher.Log(totalTorque, "totalTorque");
            Grapher.Log(at, "at");
            Grapher.Log(m_JdController.bodyPartsDict[upperLegLeft].rb.transform.localRotation.eulerAngles.x, "upperLegLeft localRotation");
            Grapher.Log(m_JdController.bodyPartsDict[upperLegRight].rb.transform.localRotation.eulerAngles.x, "upperLegRight localRotation");
            //Grapher.Log(rewardList, "rewardList");
            Grapher.Log(Convert.ToInt32(m_JdController.bodyPartsDict[lowerLegLeft].groundContact.touchingGround), "leftGroundContact");
            Grapher.Log(m_JdController.bodyPartsDict[lowerLegLeft].rb.velocity.magnitude, "Velocity");

            Debug.Log(StepCount);


        }
        
        if(externalForces) addPerturbationForce();

        if (externalBalls) {
            if (obstacle.GetComponent<GroundContact>().touchingGround)
            {
                obstacle.transform.localPosition = new Vector3(0, -100, 0);
            }
            if (Time.fixedTime >= nextInvokeTime) {
                addPerturbationObject();
                nextInvokeTime = Time.fixedTime + repeatInterval;
            }
        }

        if (runTestRoutine) {
            //Debug.Log(Time.fixedTime - episodeBeginTime);
            
            //float cycleTime = (getPhase()/(2f * Mathf.PI)) * walkCycleTime;

            groundContactListL.Add(m_JdController.bodyPartsDict[lowerLegLeft].groundContact.touchingGround ? 1 : 0);
            groundContactListR.Add(m_JdController.bodyPartsDict[lowerLegRight].groundContact.touchingGround ? 1 : 0);
            
            if(m_JdController.bodyPartsDict[lowerLegLeft].groundContact.touchingGround == true && oldTouchingGroundL == false && check1 && check2 && check3){
                if(timeOfLeftContact > episodeBeginTime) {
                    footLstart.Add(Time.fixedTime - timeOfLeftContact);
                }
                check1 = false;
                check2 = false;
                check3 = false;
                timeOfLeftContact = Time.fixedTime;
            }if(m_JdController.bodyPartsDict[lowerLegLeft].groundContact.touchingGround == false && oldTouchingGroundL == true && timeOfLeftContact > episodeBeginTime){
                footLstop.Add(Time.fixedTime - timeOfLeftContact);
                check1 = true;
            }if(m_JdController.bodyPartsDict[lowerLegRight].groundContact.touchingGround == true && oldTouchingGroundR == false && timeOfLeftContact > episodeBeginTime){
                footRstart.Add(Time.fixedTime - timeOfLeftContact);
                check2 = true;
            }if(m_JdController.bodyPartsDict[lowerLegRight].groundContact.touchingGround == false && oldTouchingGroundR == true && timeOfLeftContact > episodeBeginTime){
                footRstop.Add(Time.fixedTime - timeOfLeftContact);
                check3 = true;
            }
            oldTouchingGroundL = m_JdController.bodyPartsDict[lowerLegLeft].groundContact.touchingGround;
            oldTouchingGroundR = m_JdController.bodyPartsDict[lowerLegRight].groundContact.touchingGround;

            

        }
    }

    private double standardDeviation(IEnumerable<float> values){   
        double standardDeviation = 0;
        if (values.Any()) 
        {      
            double avg = values.Average();  // Compute the average.  
            double sum = values.Sum(d => Math.Pow(d - avg, 2));  // Perform the Sum of (value-avg)_2_2.
            standardDeviation = Math.Sqrt((sum) / (values.Count() - 1));  // Put it all together.
        }  

        return standardDeviation;
    }

    /// <summary>
    /// Returns the average velocity of all of the body parts. Using the velocity of the hips
    /// only has shown to result in more erratic movement from the limbs, // !!!!!!!!!!!!!!!!!!!!!!!!!!!! CHANGE THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    /// so using the average helps prevent this erratic movement
    /// </summary>
    Vector3 GetAvgVelocity() {
        Vector3 velSum = Vector3.zero;

        int numOfRb = 0;
        foreach (var item in m_JdController.bodyPartsList){
            numOfRb++;
            velSum += item.rb.velocity;
        }

        return velSum / numOfRb;
    }

    void addPerturbationForce() {

        Vector3 force = new Vector3(Random.Range(-1f, 1f), Random.Range(-1f, 1f), Random.Range(-1f, 1f));

        int i = Random.Range(0, m_JdController.bodyPartsList.Count);
        m_JdController.bodyPartsList[i].rb.AddForce(force * perturbationScale, ForceMode.Force);
        
        
    }

    void addPerturbationObject() {

        float angle = Random.Range(-(float)Math.PI, (float)Math.PI);
        obstacle.transform.localPosition = m_JdController.bodyPartsDict[body].rb.transform.localPosition + new Vector3((float)Math.Sin(angle), Random.Range(-0.4f, 0.4f), (float)Math.Cos(angle));
        float scale = Random.Range(1f, 2f);
        obstacle.transform.localScale = new Vector3 (1 ,1 ,1) * scale * 0.1f;
        
        Vector3 pointingVector = m_JdController.bodyPartsDict[body].rb.transform.localPosition - obstacle.transform.localPosition;

        obstacle.GetComponent<Rigidbody>().velocity = new Vector3(0, 0, 0);
        obstacle.GetComponent<Rigidbody>().mass = scale * scale * scale;
        obstacle.GetComponent<Rigidbody>().AddForce(pointingVector * Random.Range(2000f, 20000f), ForceMode.Force);

    }

    /// <summary>
    /// normalized value of the difference in avg speed vs goal walking speed.
    /// </summary>
   // public float GetMatchingVelocityReward(Vector3 velocityGoal, Vector3 actualVelocity) {
        //distance between our actual velocity and goal velocity
   //     var velDeltaMagnitude = Mathf.Clamp(Vector3.Distance(actualVelocity, velocityGoal), 0, TargetWalkingSpeed);

        //return the value on a declining sigmoid shaped curve that decays from 1 to 0
        //This reward will approach 1 if it matches perfectly and approach zero as it deviates
    //    return Mathf.Pow(1 - Mathf.Pow(velDeltaMagnitude / TargetWalkingSpeed, 2), 2);
    //}

    /// <summary>
    /// Randomises environment and agent parameters to improve ANN generalisation
    /// </summary>
    public void RandomiseDomain() {
        // Randomise Friction
        float friction = Random.Range(0.50f, 0.94f);
        ground.GetComponent<MeshCollider>().material.dynamicFriction = friction;

        // Static friction is often higher but never lower than dynamic friction
        ground.GetComponent<MeshCollider>().material.staticFriction = friction + Random.Range(0.0f, 0.04f);

        // Randomise Ground Slope
        if (groundRandEnabled) ground.transform.rotation = Quaternion.Euler(Random.Range(-1.5f, 1.5f), 0, Random.Range(-1.5f, 1.5f));


        if (stairsEnabled) {
            // Step & Stair generation
            int numOfSteps = (int)Math.Floor(8 * Math.Exp(-Random.Range(0f, 1f) * 4.15888888f) + 1); //4.605   // 1-8 steps
            numOfSteps = 2; //-------------------------------------------------------------------------------------------------------------------------
            float stairRotation = Random.Range(-5, 5);
            float stepRise = 0.1f; //Random.Range(0.0f, 0.145f);  // Random.Range(0.145f, 0.205f); //Random.Range(0.005f, 0.1f);
            float stepRun = 0.455f; //Random.Range(0.245f, 0.455f);
            float landing = 3;//Random.Range(0.8f, 4.0f);
            float stairWidth = (numOfSteps * stepRun * 2) + landing;
            Vector3 stairPosition = new Vector3(0, 0, ((stairWidth / 2f) + 5f) / ground.transform.localScale.z); //+ Random.Range(1f, 10f)
            //if(Random.Range(0f, 1f) > 0.5f) {
            //    stairPosition = new Vector3(0, 0, (Random.Range(-(landing / 2f) + 0.3f, (landing / 2f) - 0.3f)) / ground.transform.localScale.z);
            //    m_JdController.bodyPartsDict[body].rb.transform.position += new Vector3(0, stepRise * numOfSteps, 0);
            //}
            for (int i = 0; i < numOfSteps; i++) {  // Choose a random number of steps weighted towards fewer steps
                steps[i].transform.localPosition = stairPosition; // new Vector3(Random.Range(-10f, 10f), 0, Random.Range(-10f, 10f));
                steps[i].transform.localScale = new Vector3(10f / ground.transform.localScale.x, ((i + 1) * stepRise * 2), (((numOfSteps - (i + 1)) * (stepRun * 2)) + landing) / ground.transform.localScale.z);
                steps[i].transform.localRotation = Quaternion.Euler(0, stairRotation, 0);
            }
            for(int i = numOfSteps; i < 10; i++) {  // Reset step position to under tyhe world
                steps[i].transform.localPosition = new Vector3(0, -2, 0);
                steps[i].transform.localScale = new Vector3(0.1f / ground.transform.localScale.x, 0.1f, 0.1f / ground.transform.localScale.y);
                steps[i].transform.localRotation = Quaternion.Euler(0, 0, 0);
            }
        }

        if (stepEnabled) {

            float stairRotation = Random.Range(-5, 5);
            float stepRise = Random.Range(0.0f, 0.100f);  // Random.Range(0.145f, 0.205f); //Random.Range(0.005f, 0.1f);
            float stepRun = Random.Range(0.245f, 0.455f);  // Justify the choice in all of these values
            float landing = Random.Range(0.1f, 2.0f);
            float stairWidth = (1 * stepRun * 2) + landing;
            Vector3 stairPosition = new Vector3(0, 0, (stairWidth / 2f + Random.Range(0f, 4f) + 0.2f) / ground.transform.localScale.z);

            steps[0].transform.localPosition = stairPosition; // new Vector3(Random.Range(-10f, 10f), 0, Random.Range(-10f, 10f));
            steps[0].transform.localScale = new Vector3(10f / ground.transform.localScale.x, (stepRise * 2), landing / ground.transform.localScale.z);
            steps[0].transform.localRotation = Quaternion.Euler(0, stairRotation, 0);
        }
    }
}
