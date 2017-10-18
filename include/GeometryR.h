/*
Copyright (c) 2000-2002, Jelle Kok, University of Amsterdam
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the University of Amsterdam nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/*! \file Geometry.h
<pre>
<b>File:</b>          Geometry.h
<b>Project:</b>       Robocup Soccer Simulation Team: UvA Trilearn
<b>Authors:</b>       Jelle Kok
<b>Created:</b>       12/02/2001
<b>Last Revision:</b> $ID$
<b>Contents:</b>      Header file for the classes VecPosition, Geometry, Line,
Circle and Rectangle. All the member
data and member method declarations for all these classes can be
found in this file toGether with some auxiliary functions for
numeric and goniometric purposes.
<hr size=2>
<h2><b>Changes</b></h2>
<b>Date</b>             <b>Author</b>          <b>Comment</b>
12/02/2001       Jelle Kok       Initial version created
09/06/2001       Remco de Boer   Version including full documentation completed
</pre>
*/

#ifndef _GEOMETRY_
#define _GEOMETRY_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef double AngRad;  /*!< Type definition for angles in degrees. */
typedef double AngDeg;  /*!< Type definition for angles in radians. */

#define INF		10000000
#define EPS		0.000001			/*!< Value used for floating point equality tests. */
const double  UnknownDoubleValue  = -1000.0; /*!< indicates unknown double    */
const int     UnknownIntValue     = -1000;   /*!< indicates unknown int       */
const int     UnknownTime         = -20;     /*!< indicates unknown time      */
const long    UnknownMessageNr    = -30;     /*!< indicates unknown message nr*/

//#define M_PI	3.1415926535897932	/* Value of pi */
//#define PI	M_PI

class Maths
{
	public:
	// auxiliary numeric functions for determining the
	// maximum and minimum of two given double values and the Maths::Sign of a value
	static double Max     ( double d1, double d2 );
	static double Min     ( double d1, double d2 );
	static int    Sign    ( double d1            );
	static double Limit	  ( double d, double dMin, double dMax);
	static int	Round(double d);
	// auxiliary goniometric functions which enable you to
	// specify angles in degrees rather than in radians
	static AngDeg Rad2Deg ( AngRad x             );
	static AngRad Deg2Rad ( AngDeg x             );
	static double cosDeg  ( AngDeg x             );
	static double sinDeg  ( AngDeg x             );
	static double tanDeg  ( AngDeg x             );
	static AngDeg atanDeg ( double x             );
	static double atan2Deg( double x,  double y  );
	static AngDeg acosDeg ( double x             );
	static AngDeg asinDeg ( double x             );

	#ifdef WIN32
	/*! This function shall return the integral value (represented as a double)
	nearest x in the direction of the current rounding mode. The current
	rounding mode is implementation-defined.

	If the current rounding mode rounds toward negative infinity, then Maths::rint()
	shall be equivalent to floor() . If the current rounding mode rounds
	toward positive infinity, then Maths::rint() shall be equivalent to ceil().
	url: http://www.opengroup.org/onlinepubs/007904975/functions/Maths::rint.html */
	static double rint(double x) { return floor(x);	}

	/*! The Maths::drand48() function shall return non-negative,
	double-precision, floating-point values, uniformly distributed over the
	interval [0.0,1.0).
	url: http://www.opengroup.org/onlinepubs/007904975/functions/Maths::drand48.html */
	static double drand48()	{ return ((double)(rand() % 100)) / 100; }
	#endif

};

// various goniometric functions
bool   IsAngInInterval     ( AngRad ang,    AngRad angMin,    AngRad angMax );
AngRad GetBisectorTwoAngles( AngRad angMin, AngRad angMax );

/*! CoordSystem is an enumeration of the different specified coordinate systems.
The two possibilities are CARTESIAN or POLAR. These values are for instance
used in the initializing a VecPosition. The CoordSystem indicates whether
the supplied arguments represent the position in cartesian or in polar
coordinates. */
enum CoordSystemT
{
	CARTESIAN,
	POLAR
};

/******************************************************************************/
/********************   CLASS VECPOSITION   ***********************************/
/******************************************************************************/

/*! This class contains an x- and y-coordinate of a position (x,y) as member
data and methods which operate on this position. The standard arithmetic
operators are overloaded and can thus be applied to positions (x,y). It is
also possible to represent a position in polar coordinates (r,phi), since
the class contains a method to convert these into cartesian coordinates
(x,y). */
class VecPosition
{
	// private member data
	private:

	double m_x;   /*!< x-coordinate of this position */
	double m_y;   /*!< y-coordinate of this position */

	// public methods
	public:
	// constructor for VecPosition class
	VecPosition                               ( double            vx = 0,
	double            vy = 0,
	CoordSystemT      cs = CARTESIAN);

	// overloaded arithmetic operators
	VecPosition        operator -             (                                 );
	VecPosition        operator +             ( const double      &d            );
	VecPosition        operator +             ( const VecPosition &p            );
	VecPosition        operator -             ( const double      &d            );
	VecPosition        operator -             ( const VecPosition &p            );
	VecPosition        operator *             ( const double      &d            );
	VecPosition        operator *             ( const VecPosition &p            );
	VecPosition        operator /             ( const double      &d            );
	VecPosition        operator /             ( const VecPosition &p            );
	void               operator =             ( const double      &d            );
	void               operator +=            ( const VecPosition &p            );
	void               operator +=            ( const double      &d            );
	void               operator -=            ( const VecPosition &p            );
	void               operator -=            ( const double      &d            );
	void               operator *=            ( const VecPosition &p            );
	void               operator *=            ( const double      &d            );
	void               operator /=            ( const VecPosition &p            );
	void               operator /=            ( const double      &d            );
	bool               operator !=            ( const VecPosition &p            );
	bool               operator !=            ( const double      &d            );
	bool               operator ==            ( const VecPosition &p            );
	bool               operator ==            ( const double      &d            );

	// methods for producing output
	void               ToString               ( char buf[], CoordSystemT      cs = CARTESIAN);

	// Set- and Get methods for private member variables
	bool               SetX                   ( double            dX            );
	double             GetX                   (                           ) const;
	bool               SetY                   ( double            dY            );
	double             GetY                   (                           ) const;

	// Set- and Get methods for derived position information
	void               SetVecPosition         ( double            dX = 0,
	double            dY = 0,
	CoordSystemT      cs = CARTESIAN);
	double             GetDistanceTo          ( const VecPosition p             );
	VecPosition        SetMagnitude           ( double            d             );
	double             GetMagnitude           (                           ) const;
	AngRad             GetDirection           (                           ) const;
	AngRad			   GetDirectionTo         ( const VecPosition p             );

	// comparison methods for positions
	bool               IsRightOf            ( const VecPosition &p            );
	bool               IsRightOf            ( const double      &d            );
	bool               IsLeftOf             ( const VecPosition &p            );
	bool               IsLeftOf             ( const double      &d            );
	bool               IsButtomOf               ( const VecPosition &p            );
	bool               IsButtomOf               ( const double      &d            );
	bool               IsTopOf              ( const VecPosition &p            );
	bool               IsTopOf              ( const double      &d            );
	bool               IsBetweenX             ( const VecPosition &p1,
	const VecPosition &p2           );
	bool               IsBetweenX             ( const double      &d1,
	const double      &d2           );
	bool               IsBetweenY             ( const VecPosition &p1,
	const VecPosition &p2           );
	bool               IsBetweenY             ( const double      &d1,
	const double      &d2           );

	// conversion methods for positions
	VecPosition        Normalize              (                                 );
	VecPosition        Rotate                 ( AngRad            angle         );
	VecPosition        GlobalToRelative       ( VecPosition       orig,
	AngRad            ang           );
	VecPosition        RelativeToGlobal       ( VecPosition       orig,
	AngRad            ang           );
	VecPosition        GetVecPositionOnLineFraction( VecPosition  &p,
	double            dFrac         );

	// static class methods
	static VecPosition GetVecPositionFromPolar( double            dMag,
	AngRad            ang           );
	static AngRad      NormalizeAngle         ( AngRad            angle         );
	static AngRad      NormalizeAngle2PI      ( AngRad            angle         );
};

/******************************************************************************/
/*********************   CLASS GEOMETRY   *************************************/
/******************************************************************************/

/*! This class contains several static methods dealing with geometry.*/
class Geometry
{

	public:

	// geometric series
	static double GetLengthGeomSeries(double dFirst,double dRatio,double dSum   );
	static double GetSumGeomSeries   (double dFirst,double dRatio,double dLength);
	static double GetSumInfGeomSeries(double dFirst,double dRatio               );
	static double GetFirstGeomSeries (double dSum,  double dRatio,double dLength);
	static double GetFirstInfGeomSeries(double dSum,double dRatio               );

	// abc formula
	static int    AbcFormula(double a,double b, double c, double *s1, double *s2);
};

/******************************************************************************/
/********************** CLASS CIRCLE ******************************************/
/******************************************************************************/

/*!This class represents a circle. A circle is defined by one VecPosition
(which denotes the center) and its radius. */
class Circle
{
	VecPosition m_posCenter;            /*!< Center of the circle  */
	double      m_dRadius;              /*!< Radius of the circle  */

	public:
	Circle( );
	Circle( VecPosition pos, double dR );

	// Get and Set methods
	bool        SetCircle             ( VecPosition pos,
	double      dR  );
	bool        SetRadius             ( double dR       );
	double      GetRadius             (                 );
	bool        SetCenter             ( VecPosition pos );
	VecPosition GetCenter             (                 );
	double      GetCircumference      (                 );
	double      GetArea               (                 );

	// Calculate intersection points and area with other circle
	bool        IsInside              ( VecPosition pos );
	int         GetIntersectionPoints ( Circle      c,
	VecPosition *p1,
	VecPosition *p2 );
	double      GetIntersectionArea   ( Circle c        );


}  ;

/******************************************************************************/
/*********************** CLASS LINE *******************************************/
/******************************************************************************/

/*!This class contains the representation of a line. A line is defined
by the formula ay + bx + c = 0. The coefficients a, b and c are stored
and used in the calculations. */
class Line
{
public:
	// a line is defined by the formula: ay + bx + c = 0
	double m_a; /*!< This is the a coefficient in the line ay + bx + c = 0 */
	double m_b; /*!< This is the b coefficient in the line ay + bx + c = 0 */
	double m_c; /*!< This is the c coefficient in the line ay + bx + c = 0 */

public:
		VecPosition GetPointInLine(VecPosition pos1, double dDistance);
	Line( double a, double b, double c );
	Line();
	void SetLine(double a, double b, double c) { m_a = a; m_b = b; m_c = c; };


	// Get intersection points with this line
	VecPosition GetIntersection            ( Line        line                   );
	int         GetCircleIntersectionPoints( Circle      circle,
	VecPosition *posSolution1,
	VecPosition *posSolution2          );
	Line        GetPerpendicularLine       ( VecPosition pos                    );
	VecPosition GetPointOnLineClosestTo    ( VecPosition pos                    );
	double      GetDistanceWithPoint       ( VecPosition pos                    );
	bool        IsInBetween                ( VecPosition pos,
	VecPosition point1,
	VecPosition point2                 );

	// Calculate associated variables in the line
	double GetYGivenX                      ( double      x );
	double GetXGivenY                      ( double      y );
	double GetACoefficient                 (               ) const;
	double GetBCoefficient                 (               ) const;
	double GetCCoefficient                 (               ) const;
	
	double GetSlope() const { return -m_b/m_a; };

	// static methods to Make a line using an easier representation.
	static Line MakeLineFromTwoPoints      ( VecPosition pos1,
	VecPosition pos2                   );
	static Line MakeLineFromPositionAndAngle( VecPosition vec,
	AngRad angle                       );
};

/******************************************************************************/
/********************** CLASS RECTANGLE ***************************************/
/******************************************************************************/

/*!This class represents a rectangle. A rectangle is defined by two VecPositions
the one at the upper left corner and the one at the right bottom. */
class Rectangle
{
	VecPosition m_posLeftTop;     /*!< top left position of the rectangle       */
	VecPosition m_posRightBottom; /*!< bottom right position of the rectangle   */

	public:
	Rectangle                          ( VecPosition pos, VecPosition pos2 );
	Rectangle					() {};
	

	// checks whether point lies inside the rectangle
	bool        IsInside          ( VecPosition pos                   );

	// standard Get and Set methosd
	void        SetRectanglePoints( VecPosition pos1,
	VecPosition pos2                  );
	bool        SetPosLeftTop     ( VecPosition pos                   );
	VecPosition GetPosLeftTop     (                    );
	bool        SetPosRightBottom ( VecPosition pos                   );
	VecPosition GetPosRightBottom (                    );

	void        SetLeft(double x) { m_posLeftTop.SetX(x) ; };
	void        SetTop(double y) { m_posLeftTop.SetY(y) ; };
	void        SetRight(double x) { m_posRightBottom.SetX(x) ; };
	void        SetBottom(double y) { m_posRightBottom.SetY(y) ; };
};

#endif
