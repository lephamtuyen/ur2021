using UnityEngine;


public class MoveObject : MonoBehaviour
{   

    public bool moving = false ;
    public bool selected = false;

    void Update () 
    {
        if (transform.position.y < 0.7f || transform.position.y > 0.755f)
        {
            moving = false;
        }
        else
        {
            moving = true;
        }
    }
}