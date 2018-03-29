package quanto.util.test

import quanto.util.TreeSeq

// simple implementation. no complaints if the tree is invalid
case class SimpleTreeSeq[A](seq: Seq[(A,Option[A])] = Seq()) extends TreeSeq[A] {
  def toSeq = seq.map(_._1)
  def indexOf(a: A) = seq.indexWhere(_._1 == a)
  def parent(a: A) = seq.find(_._1 == a) match { case Some((_,p)) => p; case _ => None }
  def children(a: A) = seq.filter(_._2 == Some(a)).map(_._1)

  def :+(a: A, p: Option[A]) = new SimpleTreeSeq(seq :+ (a,p))
}
