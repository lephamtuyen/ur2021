using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Conveyor : MonoBehaviour
{
    public float speed = 0.05f;
    public bool start = false;
    void FixedUpdate()
    {
        if(start)
        {
            Rigidbody rb = GetComponent<Rigidbody>();
            rb.position -= transform.right *speed* Time.deltaTime;
            rb.MovePosition(rb.position + transform.right * speed *Time.deltaTime);
        }
    }
    public void moveConveyor()
    {
        start = true;
    }

    public void stopConveyor()
    {
        start = false;
    }

    public void conveyorSpeed(float speed)
    {
        speed = speed;
    }
}
